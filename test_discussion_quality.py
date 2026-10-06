import copy
import unittest
from tools.discussion_quality import audit


class DiscussionQualityTest(unittest.TestCase):
    def test_invalid_event_records_do_not_produce_a_clean_report(self):
        for malformed in ([],{'events':'not an array'},{'events':['not an event']}):
            with self.subTest(malformed=malformed),self.assertRaises(ValueError):audit(malformed)

    def test_same_opinion_repetition_is_visible_without_rejecting_agreement(self):
        label='同じ条件で小規模な試験を続ける'
        run={'events':[{'claim_id':f'C{i}','persona_id':speaker,'code':'P','label':label,
                        'utterance':label+'案を提案します。','statement':'同じ条件の試験結果を使います。[D01]',
                        'utterance_origin':'model_renderer_v3','statement_origin':'label_fallback'}for i,speaker in enumerate(['a','b','c'],1)],
             'metrics':{'dialogue_near_duplicate_rate':0,'hard_gate_pass':True}}
        before=copy.deepcopy(run);result=audit(run)
        self.assertEqual(result['same_stance_pairs_audited'],3)
        self.assertEqual(len(result['same_stance_surface_pairs']),3)
        self.assertEqual(len(result['same_stance_base_reason_pairs']),3)
        self.assertEqual(result['same_stance_base_reason_pairs'][0]['origins'],['label_fallback','label_fallback'])
        self.assertEqual(run,before)
        self.assertFalse(result['changes_existing_gates'])
        self.assertFalse(result['copying_proved'])

    def test_different_reasons_and_same_speaker_repetition_are_not_confused(self):
        run={'events':[{'claim_id':str(i),'persona_id':speaker,'code':'P','label':'試験を続ける',
              'utterance':text,'statement':text,'utterance_origin':origin}
             for i,(speaker,text,origin)in enumerate([
                 ('a','費用は予算内なので、比較の試験を続けたいです。','model_renderer_v3'),
                 ('b','効果は未確認なので、条件を限定して検証します。','model_renderer_v3_sanitized'),
                 ('a','費用は予算内なので、比較の試験を続けたいです。','template_fallback')])]}
        result=audit(run)
        self.assertEqual(result['same_stance_pairs_audited'],2)
        self.assertFalse(result['same_stance_surface_pairs'])
        self.assertEqual(result['origin_counts']['template_fallback'],1)


if __name__=='__main__':unittest.main()
