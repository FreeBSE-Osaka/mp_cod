import json
from pathlib import Path
import unittest
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stdout
import io
import sys

import cod_model as cod
from tools.general_utterance_training import cases,checks,example,item,profile_case,score,valid,PROFILES,build,balanced_training_cases,own_only_example
from tools.general_body_training import train,assistant_only_loss,exact_length_batches


class FullUtteranceTrainingTest(unittest.TestCase):
    def test_own_only_rehearsal_keeps_all_old_examples_and_frozen_evaluation_splits(self):
        curated=Path(__file__).parent/'data/general_utterance_qwen35_v2/curated.json'
        with tempfile.TemporaryDirectory()as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            args=SimpleNamespace(out=root/'parent',curated=curated,profiles=PROFILES,rehearsal=None,balanced_pairs=True)
            build(args)
            args.out=root/'next';args.own_only_rehearsal=True;build(args)
            self.assertEqual((root/'parent/valid.jsonl').read_bytes(),(root/'next/valid.jsonl').read_bytes())
            self.assertEqual((root/'parent/test.jsonl').read_bytes(),(root/'next/test.jsonl').read_bytes())
            old=[json.loads(line)for line in (root/'parent/train.jsonl').read_text().splitlines()]
            new=[json.loads(line)for line in (root/'next/train.jsonl').read_text().splitlines()]
            self.assertEqual(len(old),256);self.assertEqual(len(new),320)
            canon=lambda row:json.dumps(row,ensure_ascii=False,sort_keys=True)
            self.assertTrue({canon(row)for row in old}.issubset({canon(row)for row in new}))
            repairs=[row for row in new if '再評価せず、その立場を自然な発言' in row['messages'][0]['content']]
            self.assertEqual(len(repairs),64)
            pairs=set()
            for row in repairs:
                payload=json.loads(row['messages'][1]['content'])['items'][0]
                pairs.add((payload['speaker'],payload['move']))
                self.assertTrue({'target_claim','previous_claim','candidate_reason','perspective'}.isdisjoint(payload))
                self.assertTrue(cod.dialogue_numbers_are_grounded(json.loads(row['messages'][2]['content'])['utterances'][0]['utterance'],payload))
            self.assertEqual(len(pairs),64)
            manifest=json.loads((root/'next/manifest.json').read_text())
            self.assertEqual(manifest['own_only_train_cases'],64)
            args.out=root/'bad';args.profiles=('flexible_plain',);args.balanced_pairs=False
            with self.assertRaisesRegex(ValueError,'requires source-grounded'):build(args)

    def test_exact_padding_preserves_real_tokens_lengths_and_native_arguments(self):
        class Batch:
            def __init__(self):self.rows=[[10,11,20,21,22]+[0]*28,[30,31,32]+[0]*30]
            def __getitem__(self,key):return [row[:key[1].stop]for row in self.rows]
        class Lengths:
            def __getitem__(self,key):return [5,3]
        batch,lengths=Batch(),Lengths()
        calls=[]
        def iterator(*values,**kwargs):
            calls.append((values,kwargs));yield batch,lengths
        mx=ModuleType('mlx.core');mx.max=lambda values:SimpleNamespace(item=lambda:max(values))
        package=ModuleType('mlx');package.core=mx
        native=ModuleType('mlx_lm.tuner.trainer');native.iterate_batches=iterator
        with patch.dict(sys.modules,{'mlx':package,'mlx.core':mx,'mlx_lm.tuner.trainer':native}):
            result=list(exact_length_batches('dataset',batch_size=2,max_seq_length=768,loop=True))
        self.assertEqual(result[0][0],[[10,11,20,21,22],[30,31,32,0,0]])
        self.assertIs(result[0][1],lengths)
        self.assertEqual(calls,[(('dataset',),{'batch_size':2,'max_seq_length':768,'loop':True})])
        self.assertEqual(len(batch.rows[0]),33)

    def test_balanced_corpus_covers_each_role_move_and_all_training_topics(self):
        corpus=cases(Path(__file__).parent/'data/general_utterance_qwen35_v2/curated.json')
        selected=balanced_training_cases(corpus)
        training=[row for row in selected if row['split']=='train']
        self.assertEqual(len(training),64)
        self.assertEqual(len({(row['speaker'],row['move'])for row in training}),64)
        self.assertEqual(len({row['topic']for row in training}),10)
        self.assertEqual([row for row in selected if row['split']!='train'],
                         [row for row in corpus if row['split']!='train'])
        self.assertEqual(selected,balanced_training_cases(corpus))
        with self.assertRaisesRegex(ValueError,'every General persona/move'):
            balanced_training_cases([row for row in corpus if row['move']!='revise'])
        with tempfile.TemporaryDirectory()as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            args=SimpleNamespace(out=root/'balanced',curated=Path(__file__).parent/'data/general_utterance_qwen35_v2/curated.json',
                                 profiles=PROFILES,rehearsal=None,balanced_pairs=True)
            build(args)
            rows=[json.loads(line)for line in (args.out/'train.jsonl').read_text().splitlines()]
            counts={}
            modes={}
            for row in rows:
                payload=json.loads(row['messages'][1]['content'])['items'][0]
                key=payload['speaker'],payload['move']
                counts[key]=counts.get(key,0)+1
                mode=('source_misleading'if 'どの条件でも同じ' in payload.get('candidate_reason','')
                      else 'source_clean'if 'candidate_reason' in payload
                      else 'flexible_plain'if payload['id']=='I01'else 'structured_plain')
                modes[mode]=modes.get(mode,0)+1
            self.assertEqual(len(rows),256)
            self.assertEqual(set(counts.values()),{4})
            self.assertEqual(len(counts),64)
            self.assertEqual(modes,{mode:64 for mode in
                ('source_clean','source_misleading','flexible_plain','structured_plain')})
            args.out=root/'bad';args.rehearsal=args.curated
            with self.assertRaisesRegex(ValueError,'no extra rehearsal'):build(args)

    def test_loss_bound_excludes_padding_and_retains_all_assistant_positions(self):
        class Lengths:
            def __init__(self):
                self.values=[(2,5),(4,7)];self.at=self
            def __getitem__(self,key):
                self.asserted_column=key
                return self
            def add(self,value):
                return [(offset,end+value)for offset,end in self.values]
        lengths=Lengths()
        native=ModuleType('mlx_lm.tuner.trainer')
        native.default_loss=lambda model,batch,bounds:bounds
        with patch.dict(sys.modules,{'mlx_lm.tuner.trainer':native}):
            adjusted=assistant_only_loss('model','batch',lengths)
        self.assertEqual(adjusted,[(2,4),(4,6)])
        self.assertEqual(lengths.values,[(2,5),(4,7)])
        self.assertEqual(lengths.asserted_column,(slice(None),1))
        self.assertEqual(list(range(adjusted[0][0],adjusted[0][1]+1)),[2,3,4])

    def test_fresh_training_omits_resume_and_restores_native_globals(self):
        calls = []
        original_load = lambda *a, **k: None
        original_argv = sys.argv
        def native_train(*values,**kwargs):
            self.assertIs(kwargs['loss'],assistant_only_loss)
            if args.exact_training_padding:
                self.assertIs(kwargs['iterate_batches'],exact_length_batches)
            else:
                self.assertNotIn('iterate_batches',kwargs)
        def native_main():
            calls.append(list(sys.argv))
            self.assertNotIn('--resume-adapter-file', sys.argv)
            self.assertIn('--mask-prompt', sys.argv)
            native.train(model='unit')
        native = SimpleNamespace(load=original_load, train=native_train, main=native_main)
        package = ModuleType('mlx_lm'); package.lora = native
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(out=root/'new', model=root/'model', data=root/'data',
                                   config=root/'config.yaml', parent_adapter=None,exact_training_padding=True)
            with patch.dict(sys.modules, {'mlx_lm': package}):
                train(args)
                args.exact_training_padding=False
                train(args)
                args.parent_adapter = root/'missing.safetensors'
                with self.assertRaisesRegex(ValueError, 'existing weights file'):
                    train(args)
        self.assertEqual(len(calls), 2)
        self.assertIs(native.load, original_load)
        self.assertIs(native.train,native_train)
        self.assertIs(sys.argv, original_argv)

    def test_training_cache_cap_is_process_local_and_restored_on_failure(self):
        cache={'limit':12345,'calls':[]}
        def set_limit(value):
            old=cache['limit'];cache['limit']=value;cache['calls'].append(value);return old
        mx=ModuleType('mlx.core');mx.set_cache_limit=set_limit
        mx.get_peak_memory=lambda:0;mx.get_active_memory=lambda:0;mx.get_cache_memory=lambda:0
        package=ModuleType('mlx');package.core=mx
        original_load=lambda *a,**k:None
        def fail():raise RuntimeError('native training failed')
        original_train=lambda *a,**k:None
        lora=SimpleNamespace(load=original_load,train=original_train,main=fail)
        mlx_lm=ModuleType('mlx_lm');mlx_lm.lora=lora
        with tempfile.TemporaryDirectory()as directory:
            parent=Path(directory)/'parent.safetensors';parent.write_text('fixture')
            args=SimpleNamespace(out=Path(directory)/'out',model=Path(directory),data=Path(directory),
                                 config=Path(directory)/'config.yaml',parent_adapter=parent,mlx_cache_limit_mib=0)
            with patch.dict('sys.modules',{'mlx':package,'mlx.core':mx,'mlx_lm':mlx_lm}),redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'native training failed'):train(args)
                lora.main=lambda:None
                with patch('builtins.print',side_effect=BrokenPipeError('report failed')):
                    with self.assertRaisesRegex(BrokenPipeError,'report failed'):train(args)
        self.assertEqual(cache['calls'],[0,12345,0,12345])
        self.assertEqual(cache['limit'],12345)
        self.assertIs(lora.load,original_load)
        self.assertIs(lora.train,original_train)

    def test_v2_profiles_preserve_primary_inputs_and_cover_all_role_moves(self):
        corpus=cases(Path(__file__).parent/'data/general_utterance_qwen35_v2/curated.json')
        self.assertEqual({s:sum(row['split']==s for row in corpus)for s in ('train','valid','test')},
                         {'train':80,'valid':16,'test':24})
        speakers={row['speaker']for row in corpus}
        for speaker in speakers:
            self.assertEqual({row['move']for row in corpus if row['speaker']==speaker and row['split']=='train'},
                             {'propose','counterproposal','object','agree','improve','elaborate','maintain','revise'})
        for row in corpus:
            for profile in PROFILES:
                case=profile_case(row,profile);request=example(case)
                with self.subTest(case=row['case'],profile=profile):
                    raw=request['messages'][2]['content']
                    self.assertTrue(score(raw,case)['study_valid'])
                    self.assertIn('{"utterances":[{"id":',request['messages'][0]['content'])
                    self.assertEqual(cod.RENDERER_JSON_RULE in request['messages'][0]['content'],
                                     profile!='source_grounded')
                    payload=json.loads(request['messages'][1]['content'])['items'][0]
                    self.assertEqual(payload['evidence'],row['evidence'])
                    self.assertNotIn(case['utterance'],request['messages'][1]['content'])
                    self.assertEqual('perspective'in payload,profile=='source_grounded')
                    self.assertEqual('candidate_reason'in payload,profile=='source_grounded')
                    if row['move']=='revise':
                        if profile=='structured_plain':
                            self.assertEqual(payload['previous_choice'],row['previous_claim'])
                        else:
                            self.assertEqual(payload['previous_claim'],row['previous_claim'])
                            self.assertTrue(payload['previous_choice'].endswith('_PREVIOUS'))
                        self.assertNotEqual(row['previous_claim'],payload['selected_claim'])

    def test_v2_source_coverage_is_not_runtime_syntax_and_never_enters_prompt(self):
        corpus=cases(Path(__file__).parent/'data/general_utterance_qwen35_v2/curated.json')
        row=next(x for x in corpus if x['case']=='audio_cue:1')
        raw=example(row)['messages'][2]['content'].replace('未検証','検証済み')
        result=score(raw,row)
        self.assertTrue(result['direct_valid'])
        self.assertFalse(result['study_valid'])
        self.assertFalse(result['source_coverage'])
        payload=json.loads(example(row)['messages'][1]['content'])['items'][0]
        self.assertNotIn('required',payload)
        self.assertNotIn('forbidden',payload)

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
