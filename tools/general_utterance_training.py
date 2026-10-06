#!/usr/bin/env python3.11
"""Full-utterance corpus and direct-generation checks using the runtime validators."""
import argparse
import json
from pathlib import Path
import random
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cod_model as cod
from tools.general_body_training import guarded_job, sha, train


def cases(path):
    document = json.loads(Path(path).read_text())
    topics = document["topics"]
    id_width = 2 if document.get('schema_version') == 2 else 3
    roster = cod.load_domains()["general"]["personas"]
    result, seen = [], set()
    for n, topic in enumerate(topics):
        if topic["id"] in seen or topic["split"] not in {"train", "valid", "test"}:
            raise ValueError("duplicate topic or invalid split")
        seen.add(topic["id"])
        for i, row in enumerate(topic["examples"]):
            persona = roster[(n + i) % len(roster)]
            renderer_id = row.get("id") or (f"R1:{topic['id'].upper()}|ALTERNATIVE:{persona['id']}"
                           if row['move'] in {'maintain','revise'} else f"C{n*8+i+1:0{id_width}d}")
            result.append({**row, "case": f"{topic['id']}:{i+1}", "topic": topic["id"],
                "split": topic["split"], "speaker": persona["name"], "persona": persona,
                "evidence": topic["evidence"], "id": renderer_id})
    return result


PROFILES = ('source_grounded', 'flexible_plain', 'structured_plain')


def profile_case(case, profile):
    if profile not in PROFILES:
        raise ValueError('unknown renderer profile')
    result = {**case, 'renderer_profile': profile}
    if profile == 'structured_plain':
        result['id'] = f"{case['topic'].upper()}:{case['persona']['id']}:{case['move']}"
        result['utterance'] = case.get('structured_utterance', case['utterance'])
    elif profile == 'flexible_plain':
        result['id'] = 'I01'
    return result


def item(case, *, misleading=False):
    profile = case.get('renderer_profile', 'source_grounded')
    flexible = profile != 'structured_plain'
    phase = "reconciliation" if case["move"] in {"maintain", "revise"} else "event"
    payload = {"phase": phase, "move": case["move"], "evidence": case["evidence"]}
    if phase == "event":
        payload.update(own_claim=case["claim"], target_claim=case.get("target"))
    else:
        payload.update(selected_claim=case["claim"], previous_choice=case.get('previous_claim',
                       "PREVIOUS" if case["move"]=="revise" else "CURRENT"),
                       alternatives=case.get('alternatives', [case["claim"]]))
        if case.get('previous_claim') and profile != 'structured_plain':
            payload['previous_claim'] = case['previous_claim']
            payload['previous_choice'] = case['topic'].upper() + ('_PREVIOUS' if case['move']=='revise' else '_CURRENT')
    reason = case.get('candidate_reason', f"{case['claim']}。{case['evidence'][0]} 根拠は[D01]です。")
    if misleading:
        reason = "効果は確認済みで、どの条件でも同じ結果になる。根拠は[D01]です。"
    context = cod.source_renderer_context({"target": {"statement": reason, "statement_origin": "model"}},
                                         case["persona"], profile)
    return {"id": case["id"], **payload, "speaker": case["speaker"],
            "speech_act": cod.renderer_move_instruction(case["move"], flexible=flexible), **context}


def example(case, *, misleading=False):
    profile = case.get('renderer_profile', 'source_grounded')
    system = cod.renderer_system(None, flexible=profile != 'structured_plain',
                                 source_grounded=profile == 'source_grounded')
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps({"items": [item(case, misleading=misleading)]}, ensure_ascii=False)},
        {"role": "assistant", "content": json.dumps({"utterances": [{"id": case["id"], "utterance": case["utterance"]}]}, ensure_ascii=False)}]}


def checks(text, case):
    normalized, reason = cod.validate_dialogue_move(text, case["move"], case["claim"],
                                                   flexible=case.get('renderer_profile') != 'structured_plain')
    return {"utterance": normalized, "syntax_move": normalized is not None,
        "aligned": bool(normalized) and cod.dialogue_is_aligned(normalized, case["claim"], []),
        "grounded": bool(normalized) and cod.dialogue_numbers_are_grounded(normalized, item(case)),
        "reaction": bool(normalized) and (not case.get("target") or case["move"] not in {"object","counterproposal","agree","improve"}
            or cod.reaction_is_aligned(normalized,case["claim"],case["target"],case["move"])),
        "not_competing": bool(normalized) and (not case.get("target") or case["move"] in {"agree","improve","elaborate"}
            or not cod.dialogue_selects_competing_claim(normalized,[case["target"]])),
        "not_mechanical": bool(normalized) and not cod.is_mechanical_utterance(normalized), "reason": reason}


def valid(check):
    return all(check[key] for key in ("syntax_move","aligned","grounded","reaction","not_competing","not_mechanical"))


def score(raw, case):
    values, warning = cod.parse_renderer_utterances(cod.parse_json_object(raw), [case["id"]])
    check = checks(values.get(case["id"]), case)
    result = {"strict_schema": warning is None, "schema_warning": warning,
              "checks": check, "direct_valid": warning is None and valid(check)}
    if 'required' in case or 'forbidden' in case:
        # ponytail: frozen lexical fact checks supplement, not replace, semantic review.
        candidate = values.get(case['id'])
        text = re.sub(r'\s+', ' ', candidate) if isinstance(candidate, str) else ''
        missing = [pattern for pattern in case.get('required', []) if not re.search(pattern, text)]
        forbidden = [pattern for pattern in case.get('forbidden', []) if re.search(pattern, text)]
        result.update(source_coverage=not missing, source_errors=bool(forbidden),
                      missing_facts=missing, forbidden_assertions=forbidden,
                      study_valid=result['direct_valid'] and not missing and not forbidden)
    return result


def build(args):
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("dataset output already contains files")
    original=cases(args.curated)
    profiles=getattr(args,'profiles',('source_grounded',))
    if len(set(profiles))!=len(profiles):
        raise ValueError('duplicate renderer profile')
    corpus=[profile_case(case,profile) for case in original for profile in profiles]
    for case in corpus:
        gold=score(json.dumps({'utterances':[{'id':case['id'],'utterance':case['utterance']}]},ensure_ascii=False),case)
        if not gold.get('study_valid',gold['direct_valid']):
            raise ValueError(f"invalid authored full utterance {case['case']}: {checks(case['utterance'],case)}")
    claims={split:{row['claim'] for row in corpus if row['split']==split} for split in ('train','valid','test')}
    if any(claims[a] & claims[b] for a,b in (('train','valid'),('train','test'),('valid','test'))):
        raise ValueError('claim crosses topic splits')
    args.out.mkdir(parents=True,exist_ok=True)
    rehearsal=[]
    if getattr(args,'rehearsal',None):
        rehearsal=[case for case in cases(args.rehearsal) if case['split']=='train']
        if any(case['claim'] in claims['valid']|claims['test'] for case in rehearsal):
            raise ValueError('rehearsal claim overlaps development or test')
    counts,hashes={},{}
    for split in ('train','valid','test'):
        rows=[]
        for case in corpus:
            if case['split']==split:
                rows.append(example(case))
                if split=='train' and case['renderer_profile']=='source_grounded':
                    rows.append(example(case,misleading=True))
        if split=='train':
            for case in rehearsal:
                rows.extend((example(case),example(case,misleading=True)))
        random.Random(20261006).shuffle(rows)
        p=args.out/(split+'.jsonl');cod.write_jsonl(p,rows);counts[split]=len(rows);hashes[split]=sha(p)
    cod.write_json(args.out/'manifest.json',{'schema_version':1,'counts':counts,'hashes':hashes,'curated_sha256':sha(args.curated),
        'system':cod.renderer_system(None,flexible=True,source_grounded=True),'training_enable_thinking':False,
        'profiles':list(profiles),'rehearsal_train_cases':len(rehearsal),
        'systems':{profile:example(profile_case(original[0],profile))['messages'][0]['content'] for profile in profiles},
        'rehearsal_sha256':sha(args.rehearsal) if getattr(args,'rehearsal',None) else None,
        'scope':'full utterance only; Base decisions never trained','misleading_candidate_rehearsal':True,'promotion_allowed':False})
    print(json.dumps(counts))


def evaluate(args):
    if args.out.exists():
        raise ValueError('evaluation output already exists')
    from mlx_lm import load,generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.utils import load_adapters,remove_lora_layers
    import mlx.core as mx
    corpus=[profile_case(case,profile) for case in cases(args.curated) if case['split']==args.split
            for profile in getattr(args,'profiles',('source_grounded',))]
    model,tokenizer=load(str(args.model));model.eval()
    before=None
    if args.check_adapter_isolation:
        messages=example(corpus[0])['messages'][:2]
        probe=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        before=generate(model,tokenizer,prompt=probe,max_tokens=220,sampler=make_sampler(temp=0),verbose=False)
    if args.adapter:
        model=load_adapters(model,str(args.adapter));model.eval()
    rows=[]
    result={'curated_sha256':sha(args.curated),'validator_sha256':sha(cod.__file__),'evaluator_sha256':sha(__file__),
        'model':str(args.model),'adapter':str(args.adapter) if args.adapter else None,
        'weights_sha256':sha(args.adapter/'adapters.safetensors') if args.adapter else None,'results':rows,'promotion_allowed':False}
    for case in corpus:
        messages=example(case)['messages'][:2]
        prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        mx.random.seed(20261006);started=time.perf_counter()
        raw=generate(model,tokenizer,prompt=prompt,max_tokens=220,sampler=make_sampler(temp=0),verbose=False)
        row={'case':case['case'],'profile':case['renderer_profile'],'split':case['split'],'speaker':case['speaker'],'claim':case['claim'],'move':case['move'],
             'request':messages,'raw':raw,'seconds':time.perf_counter()-started,**score(raw,case)}
        rows.append(row);cod.write_json(args.out,result);mx.clear_cache()
        print(case['case'],row['direct_valid'],row['checks']['utterance'],flush=True)
    result['summary']={'total':len(rows),'strict_schema':sum(row['strict_schema'] for row in rows),'direct_valid':sum(row['direct_valid'] for row in rows)}
    if any('study_valid' in row for row in rows):
        result['summary']['study_valid']=sum(row.get('study_valid',row['direct_valid']) for row in rows)
    if before is not None:
        model=remove_lora_layers(model);model.eval()
        after=generate(model,tokenizer,prompt=probe,max_tokens=220,sampler=make_sampler(temp=0),verbose=False)
        result['adapter_isolation']={'base_before':before,'base_after':after,'identical':before==after,'scope':'one deterministic utterance,not all-input structural proof'}
    cod.write_json(args.out,result);print(json.dumps(result['summary']))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('build','train','evaluate'):
        p=sub.add_parser(name);p.add_argument('--out',type=Path,required=True)
        p.add_argument('--curated',type=Path,default=Path('data/general_utterance_qwen35_v1/curated.json'))
        p.add_argument('--min-free-gib',type=float,default=20);p.add_argument('--resource-check-seconds',type=float,default=2)
        if name!='build':p.add_argument('--model',type=Path,required=True)
        if name!='train':p.add_argument('--profiles',nargs='+',choices=PROFILES,default=['source_grounded'])
        if name=='build':p.add_argument('--rehearsal',type=Path)
        if name=='train':
            p.add_argument('--data',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--parent-adapter',type=Path,required=True)
            p.add_argument('--mlx-cache-limit-mib',type=int)
        if name=='evaluate':
            p.add_argument('--adapter',type=Path);p.add_argument('--split',choices=('valid','test'),default='valid');p.add_argument('--check-adapter-isolation',action='store_true')
    args=parser.parse_args()
    if args.command=='evaluate' and args.check_adapter_isolation and not args.adapter:
        parser.error('adapter isolation requires an adapter')
    return guarded_job(args,{'build':build,'train':train,'evaluate':evaluate}[args.command])


if __name__=='__main__':
    raise SystemExit(main())
