import json
from pathlib import Path
import unittest

import cod_model as cod
from tools.general_utterance_training import cases,checks,example,item,score,valid


class FullUtteranceTrainingTest(unittest.TestCase):
    def test_authored_splits_cover_all_personas_and_moves_without_cross_split_claims(self):
        corpus=cases(Path(__file__).parent/'data/general_utterance_qwen35_v1/curated.json')
        self.assertEqual({s:sum(row['split']==s for row in corpus) for s in ('train','valid','test')},
                         {'train':72,'valid':24,'test':24})
        expected={p['name'] for p in cod.load_domains()['general']['personas']}
        claims={s:{row['claim'] for row in corpus if row['split']==s} for s in ('train','valid','test')}
        for a,b in (('train','valid'),('train','test'),('valid','test')):
            self.assertFalse(claims[a]&claims[b])
        for split in claims:
            self.assertEqual({row['speaker'] for row in corpus if row['split']==split},expected)
        for speaker in expected:
            self.assertEqual({row['move'] for row in corpus if row['split']=='train' and row['speaker']==speaker},
                             {'propose','counterproposal','object','agree','improve','elaborate','maintain','revise'})
        for row in corpus:
            with self.subTest(case=row['case']):
                self.assertTrue(valid(checks(row['utterance'],row)))
                ex=example(row)
                request=json.loads(ex['messages'][1]['content'])['items'][0]
                self.assertEqual(request['id'],row['id'])
                self.assertEqual(request['perspective'],{k:row['persona'][k] for k in ('worldview','utility','loss')})
                self.assertNotIn(row['utterance'],ex['messages'][1]['content'])
                if row['move'] in {'maintain','revise'}:
                    self.assertTrue(row['id'].startswith('R1:'))
                self.assertEqual(json.loads(ex['messages'][2]['content'])['utterances'][0]['id'],row['id'])

    def test_wrong_schema_and_invented_numbers_are_not_repaired_into_success(self):
        row=next(x for x in cases(Path(__file__).parent/'data/general_utterance_qwen35_v1/curated.json') if x['move']=='elaborate')
        raw=json.dumps({'utterances':[{'id':row['id'],'utterance':row['utterance']}]},ensure_ascii=False)
        self.assertTrue(score(raw,row)['direct_valid'])
        wrong=json.dumps([{'id':row['id'],'utterance':row['utterance']}],ensure_ascii=False)
        self.assertFalse(score(wrong,row)['strict_schema'])
        bad=raw.replace('4800','9999')
        self.assertFalse(score(bad,row)['direct_valid'])
        mixed=item(row,misleading=True)
        self.assertIn('どの条件でも同じ',mixed['candidate_reason'])
        self.assertEqual(mixed['evidence'],row['evidence'])


if __name__=='__main__':
    unittest.main()
