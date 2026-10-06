import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from tools.verifier_consistency import audit_consistency

class VerifierConsistencyTest(unittest.TestCase):
    def test_supported_claim_cannot_disagree_with_its_own_total(self):
        statement='AさんとBさんに1枚ずつ発行するなら、合計の発行枚数は1枚になる。'
        # The real failure uses a longer noun phrase; both basic and actual forms must be covered.
        for text in (statement,'AさんとBさんに1枚ずつなら合計1枚になる。'):
            result=audit_consistency(text,{'verdict':'SUPPORTED','reason':'2人に各1枚ずつなので合計2枚となる。'})
            self.assertTrue(result['requires_review'])

    def test_equal_totals_and_different_units_do_not_become_proof(self):
        for statement,reason in (('合計2枚です。','合計２枚です。'),('総数2人です。','合計2名です。'),
                                 ('合計2枚です。','合計2人です。')):
            result=audit_consistency(statement,{'verdict':'SUPPORTED','reason':reason})
            self.assertFalse(result['requires_review'])
            self.assertFalse(result['automatic_approval_allowed'])
            self.assertFalse(result['semantic_support_proved'])

    def test_explicit_equations_are_recomputed_as_fractions(self):
        for statement in ('1+1=1','４０−２５＝２５','2×3=5','1/3+1/3=1/3','1/0=2'):
            self.assertTrue(audit_consistency(statement,{'verdict':'SUPPORTED','reason':'確認しました。'})['requires_review'])
        for statement in ('1+1=2','４０−２５＝１５','2×3=6','1/3+1/3=2/3'):
            self.assertFalse(audit_consistency(statement,{'verdict':'SUPPORTED','reason':'確認しました。'})['requires_review'])

    def test_false_equation_as_rejected_target_is_not_reinterpreted_as_approval(self):
        verifier={'verdict':'CONTRADICTED','reason':'1+1=2なので、対象の式は間違いです。'}
        before=copy.deepcopy(verifier)
        result=audit_consistency('1+1=1',verifier)
        self.assertFalse(result['requires_review']);self.assertEqual(verifier,before)
        self.assertTrue(audit_consistency('比較します。',{'verdict':'UNVERIFIED','reason':'40-25=25です。'})['requires_review'])
        for reason in ('1+1=1という主張は誤りです。1+1=2が正しいです。',
                       '1+1=1ではなく、1+1=2です。'):
            self.assertFalse(audit_consistency('1+1=1',{'verdict':'CONTRADICTED','reason':reason})['requires_review'])

    def test_multiple_totals_are_not_forced_into_one_scope(self):
        result=audit_consistency('元は合計1枚、追加後は合計2枚です。',{'verdict':'SUPPORTED','reason':'追加後は合計2枚です。'})
        self.assertFalse(result['requires_review'])

    def test_cli_audits_saved_output_without_overwriting_inputs_or_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'probe.json';out=root/'audit.json'
            raw=json.dumps({'verdict':'SUPPORTED','reason':'合計2枚です。'},ensure_ascii=False)
            document={'rows':[{'case':'wrong_total','raw':raw,'request':[
                {'role':'system','content':'test'},
                {'role':'user','content':json.dumps({'statement':'合計1枚です。'},ensure_ascii=False)}]}]}
            source.write_text(json.dumps(document,ensure_ascii=False),encoding='utf-8')
            original=source.read_bytes()
            command=[sys.executable,str(Path(__file__).parent/'tools/verifier_consistency.py'),str(source),'--out']
            first=subprocess.run(command+[str(out)],capture_output=True,text=True)
            self.assertEqual(first.returncode,0,first.stderr)
            audit=json.loads(out.read_text());saved=out.read_bytes()
            self.assertEqual(audit['review_count'],1)
            self.assertFalse(audit['automatic_approval_allowed'])
            for target in (source,out):
                self.assertNotEqual(subprocess.run(command+[str(target)],capture_output=True).returncode,0)
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(out.read_bytes(),saved)

if __name__=='__main__':unittest.main()
