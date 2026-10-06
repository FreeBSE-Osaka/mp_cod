#!/usr/bin/env python3.11
"""Read-only diagnostics for same-opinion repetition; never changes discussion gates."""
import argparse
from collections import Counter
from difflib import SequenceMatcher
import itertools
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cod_model as cod
from tools.general_body_training import sha


def normalize(text):
    return re.sub(r'[^\w]', '', re.sub(r'D\d{2,}', '', text).casefold())


def surface_overlap(left, right):
    a,b=normalize(left),normalize(right)
    return SequenceMatcher(None,a,b,autojunk=False).ratio() if a and b else 0.0


def speech_body(text):
    prefixes={prefix for variants in (cod.FLEXIBLE_MOVE_PREFIXES,cod.MOVE_UTTERANCE_PREFIXES)
              for values in variants.values()for prefix in values}
    for prefix in sorted(prefixes,key=len,reverse=True):
        if text.startswith(prefix):return text[len(prefix):]
    return text


def repeated_sentences(text):
    if not isinstance(text,str):return []
    seen={}
    # ponytail: exact long sentences only; paraphrases and intentional emphasis need review.
    for sentence in re.split(r'[。！？!?]+',speech_body(text)):
        key=normalize(sentence)
        if len(key)>=12:
            entry=seen.setdefault(key,{'sentence':sentence.strip(),'occurrences':0})
            entry['occurrences']+=1
    return [entry for entry in seen.values()if entry['occurrences']>1]


def audit(run):
    if not isinstance(run,dict) or not isinstance(run.get('events'),list):
        raise ValueError('expected an event-debate run with an events array')
    if any(not isinstance(row,dict) for row in run['events']):
        raise ValueError('event records must be objects')
    labels={row['code']:row.get('label','')for row in run['events']if row.get('code')}
    labels.update({row['code']:row['label']for row in run.get('ledger_snapshot',{}).get('claim_catalog',[])})
    records=[{'id':row.get('claim_id',f'C{i}'),'phase':'event','persona':row.get('persona_id'),
              'stance':row.get('code'),'label':row.get('label',''),
              'utterance':row.get('utterance',''),'statement':row.get('statement',''),
              'statement_origin':row.get('statement_origin','unknown'),
              'origin':row.get('utterance_origin','unknown')}for i,row in enumerate(run['events'],1)]
    for round_ in run.get('reconciliation',[]):
        for pair,votes in round_.get('votes',{}).items():
            for persona,row in votes.items():
                records.append({'id':f"R{round_['round']}:{pair}:{persona}",
                    'phase':f"reconciliation:{round_['round']}:{pair}",'persona':persona,
                    'stance':row.get('choice'),'label':labels.get(row.get('choice'),''),
                    'utterance':row.get('utterance',''),'statement':row.get('statement',''),
                    'statement_origin':row.get('statement_origin','unknown'),
                    'origin':row.get('utterance_origin','unknown')})
    origins=Counter(row['origin']for row in records)
    label_like=[{'id':row['id'],'persona':row['persona'],'origin':row['origin'],
                 'overlap':round(surface_overlap(speech_body(row['utterance']),row['label']),4)}
                for row in records if row['label'] and isinstance(row['utterance'],str)
                and surface_overlap(speech_body(row['utterance']),row['label'])>=0.8]
    within_repetitions=[{'id':row['id'],'persona':row['persona'],'origin':row['origin'],'repeated_sentences':repeats}
                        for row in records if (repeats:=repeated_sentences(row['utterance']))]
    compared,utterance_pairs,reason_pairs=0,[],[]
    # ponytail: quadratic per-stage audit; bucket by stance if very long transcripts need it.
    for left,right in itertools.combinations(records,2):
        if not left['persona'] or not right['persona'] or left['persona']==right['persona']:
            continue
        if not left['stance'] or left['phase']!=right['phase'] or left['stance']!=right['stance']:
            continue
        compared+=1
        for field,destination in [('utterance',utterance_pairs),('statement',reason_pairs)]:
            a,b=left[field],right[field]
            if not isinstance(a,str)or not isinstance(b,str):continue
            ratio=surface_overlap(a,b)
            if ratio>=0.9:
                destination.append({'left':left['id'],'right':right['id'],'phase':left['phase'],
                                    'stance':left['stance'],'overlap':round(ratio,4),
                                    'origins':[left['statement_origin'],right['statement_origin']]
                                              if field=='statement' else [left['origin'],right['origin']]})
    return {'schema_version':1,'utterances':len(records),'origin_counts':dict(origins),
        'statement_origin_counts':dict(Counter(row['statement_origin']for row in records)),
        'same_stance_pairs_audited':compared,'same_stance_surface_pairs':utterance_pairs,
        'same_stance_base_reason_pairs':reason_pairs,'label_like_utterances':label_like,
        'within_utterance_repetitions':within_repetitions,
        'missing_speaker_or_stance':sum(not row['persona']or not row['stance']for row in records),
        'diagnostic_only':True,'changes_existing_gates':False,'copying_proved':False,
        'policy':'Agreement and shared evidence are legitimate. High textual overlap, label-like speech or repeated long sentences prompt review; they are not proof of copying or factual correctness and do not reject intentional emphasis.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.out and args.out.exists():parser.error('output already exists')
    try:
        result=audit(json.loads(args.run.read_text()))
    except (ValueError,TypeError,KeyError) as error:
        parser.error(str(error))
    result.update(input=str(args.run),input_sha256=sha(args.run),auditor_sha256=sha(__file__))
    if args.out:cod.write_json(args.out,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
