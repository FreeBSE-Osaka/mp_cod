#!/usr/bin/env python3.11
"""Read-only arithmetic and explanation consistency checks; never approve a claim."""
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cod_model as cod
from tools.general_body_training import sha

NUMBER=r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)'
QUANTITY_UNITS=r'パーセント|万円|時間|分|秒|人|名|枚|台|件|個|円|kg|g|mL|L|%'

def normalized(text):
    return unicodedata.normalize('NFKC',text).translate(str.maketrans({'×':'*','÷':'/','−':'-'}))

def total_quantities(text):
    quantities=defaultdict(set)
    if not isinstance(text,str):return quantities
    noun=r'(?:\s*の?[ぁ-んァ-ヶ一-龠]{1,12}?(?:は|が))?'
    for number,unit in re.findall(r'(?:合計|総数|総計)'+noun+r'\s*(?:は|が)?\s*('+NUMBER+r')\s*('+QUANTITY_UNITS+r')',normalized(text)):
        quantities[{'名':'人','パーセント':'%'}.get(unit,unit)].add(Fraction(number))
    return quantities

def equation_issues(text,field):
    if not isinstance(text,str):return []
    issues=[]
    # ponytail: explicit numeric equations only; natural-language operand/subject extraction remains unproved.
    expression=NUMBER+r'(?:\s*[+*/-]\s*'+NUMBER+r')+'
    pattern=r'(?<![\w.])('+expression+r')\s*=\s*('+NUMBER+r'(?:\s*/\s*'+NUMBER+r')?)(?![\d.])'
    compact=normalized(text)
    for match in re.finditer(pattern,compact):
        # An explicitly rejected equation is a quoted error, not the auditor's asserted calculation.
        if re.match(r'\s*(?:という(?:式|主張|記述|計算)?|と(?:いう|言う)(?:式|主張|記述|計算)?)?\s*(?:は|が)?\s*(?:誤り|間違い|成立しない|ではなく)',compact[match.end():]):
            continue
        left,right=match.groups()
        try:
            computed=Fraction(cod.calculate(left).split('=')[0].strip())
            claimed=Fraction(cod.calculate(right).split('=')[0].strip())
        except (SyntaxError,ValueError,ZeroDivisionError,OverflowError) as error:
            issues.append({'kind':'equation_rejected','field':field,'expression':left,'claimed':right,'error':str(error)})
            continue
        if computed!=claimed:
            issues.append({'kind':'equation_mismatch','field':field,'expression':left,'computed':str(computed),'claimed':str(claimed)})
    return issues

def audit_consistency(statement,verifier):
    if not isinstance(statement,str)or not isinstance(verifier,dict):
        raise ValueError('expected a statement string and a verifier object')
    verdict=verifier.get('verdict')
    if verdict not in ('SUPPORTED','CONTRADICTED','UNVERIFIED')or not isinstance(verifier.get('reason'),str):
        raise ValueError('invalid verifier verdict/reason')
    reason=verifier['reason']
    issues=equation_issues(reason,'reason')
    if verdict=='SUPPORTED':
        issues.extend(equation_issues(statement,'statement'))
        claimed,explained=total_quantities(statement),total_quantities(reason)
        for unit in claimed.keys()&explained.keys():
            # Only one literal total per text/unit: competing scopes and multiple totals stay for human review.
            if len(claimed[unit])==len(explained[unit])==1 and claimed[unit]!=explained[unit]:
                issues.append({'kind':'supported_total_disagrees','unit':unit,
                    'statement_total':str(next(iter(claimed[unit]))),'reason_total':str(next(iter(explained[unit])))})
    return {'issues':issues,'requires_review':bool(issues),'automatic_approval_allowed':False,
            'semantic_support_proved':False,'policy':'No detected inconsistency is not proof of correctness. Total matching does not bind subjects or conditions.'}

def audit_saved(path):
    document=json.loads(path.read_text());rows=[]
    if not isinstance(document,dict)or not isinstance(document.get('rows'),list):raise ValueError('expected saved verifier rows')
    for row in document['rows']:
        request=json.loads(row['request'][1]['content'])
        verifier=cod.parse_json_object(row['raw'])
        result=audit_consistency(request['statement'],verifier)
        rows.append({'case':row['case'],'raw_sha256':hashlib.sha256(row['raw'].encode()).hexdigest(),**result})
    return {'scope':'saved-output rejection diagnostics only; no new model generation or score rewriting',
            'input_sha256':sha(path),'auditor_sha256':sha(__file__),'rows':rows,
            'review_count':sum(row['requires_review']for row in rows),'original_output_unchanged':True,
            'promotion_allowed':False,'automatic_approval_allowed':False}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.out and args.out.exists():parser.error('new audit output required')
    result=audit_saved(args.input)
    if args.out:
        with args.out.open('x',encoding='utf-8') as stream:
            stream.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'rows':len(result['rows']),'requires_review':result['review_count'],'approval':False}))

if __name__=='__main__':main()
