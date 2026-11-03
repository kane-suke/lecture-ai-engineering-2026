"""実測した2条件の結果ZIPから提出用の比較表を作る。"""
import argparse
import json
from pathlib import Path
import zipfile


def load(path):
    with zipfile.ZipFile(path) as z:
        return {name: json.loads(z.read(name)) for name in
                ('result.json', 'eval_before.json', 'eval_after.json', 'gold.json')}


def rate(value):
    return f'{value:.1%}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('changed', type=Path)
    parser.add_argument('--out', type=Path, default=(Path(__file__).resolve().parent.parent / 'lectures' / '02' if Path(__file__).parent.name == 'scripts' else Path(__file__).parent / 'repository' / 'lectures' / '02'))
    args = parser.parse_args()
    datasets = [load(args.baseline), load(args.changed)]
    a, b = [d['result.json'] for d in datasets]
    assert a['executed_training_myself'] and b['executed_training_myself'], '本人の実行結果を使用してください'
    changes = {key: (a['config'].get(key), b['config'].get(key)) for key in set(a['config']) | set(b['config']) if a['config'].get(key) != b['config'].get(key)}
    assert changes == {'num_train_epochs': (20, 10)}, f'想定外の条件差: {changes}'
    assert a['versions'] == b['versions'], 'ライブラリのバージョンが異なります'
    assert datasets[0]['gold.json'] == datasets[1]['gold.json'], '評価の正解が異なります'
    questions = []
    for d in datasets:
        r = d['result.json']
        assert r.get('final_loss') is not None and r.get('training_seconds', 0) > 0
        for stage in ['before', 'after']:
            e = d[f'eval_{stage}.json']
            assert e['summary'] == r[stage]
            assert len(e['rows']) == r[stage]['n'] == 21
            for metric, field in [('format_rate', 'format_ok'), ('content_rate', 'content_ok')]:
                actual = sum(row[field] for row in e['rows']) / 21
                assert abs(actual - r[stage][metric]) < 1e-12, f'{metric}の集計不一致'
        questions.append([row['question'] for row in d['eval_after.json']['rows']])
    assert questions[0] == questions[1], '評価質問の順序または内容が異なります'
    args.out.mkdir(parents=True, exist_ok=True)
    for label, d in zip(['epochs_20', 'epochs_10'], datasets):
        target = args.out / 'results' / label
        target.mkdir(parents=True, exist_ok=True)
        for name, value in d.items():
            (target / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    table = ['| 条件 | 学習前format_rate | 学習後format_rate | 学習前content_rate | 学習後content_rate | 最終loss | 学習時間（秒） |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in [a, b]:
        table.append(f'| {r["config"]["num_train_epochs"]}エポック | {rate(r["before"]["format_rate"])} | {rate(r["after"]["format_rate"])} | {rate(r["before"]["content_rate"])} | {rate(r["after"]["content_rate"])} | {r["final_loss"]:.6f} | {r["training_seconds"]:.2f} |')
    failures = [row for row in datasets[0]['eval_after.json']['rows'] if not row['both_ok']]
    detail = '\n'.join(f'- 質問：{row["question"]}\n  - 出力：`{row["raw"].replace(chr(10), " ")}`\n  - 形式：{row["format_ok"]}、内容：{row["content_ok"]}、正解needs_review：{row["gold_needs_review"]}' for row in failures) or '- 21問では形式・内容とも不正解となる質問はなかった。'
    knowledge = '\n'.join(f'- {row["質問"]}：{row["学習前の分類"]} → {row["学習後の分類"]}' for row in a['knowledge'])
    text = f'''# 第2回 解答

## 演習

- モデル：`{a['config']['base_model']}`、revision：`{a['config']['revision']}`。
- 設定：seed={a['config']['seed']}、LoRA r={a['config']['lora_r']}、alpha={a['config']['lora_alpha']}、lr={a['config']['lr']}、20エポック、量子化なし。生成はdo_sample=False。
- 自分で学習を実行したか：はい（executed_training_myself=True）。
- 学習前：format_rate={rate(a['before']['format_rate'])}、content_rate={rate(a['before']['content_rate'])}。
- 学習後：format_rate={rate(a['after']['format_rate'])}、content_rate={rate(a['after']['content_rate'])}。
- 退行した質問：{json.dumps(a['regressed'], ensure_ascii=False)}。

### 学習後に残った失敗

{detail}

### 知識を問う質問の結果

{knowledge}

## 宿題

- 取り組んだレベル：基礎。
- 変更点：CONFIGのnum_train_epochsを20から10に変更した。学習前のモデルからそれぞれ実行し、評価質問・正解・seed・モデルrevision・生成条件・その他の学習設定をそろえた。

{chr(10).join(table)}

最終lossは最終ログ区間のTraining Loss（logging_steps=10）であり、全学習期間の平均train_lossとは区別した。学習時間はtrainer.train()の前後で測定し、モデル取得・評価・保存の時間を含めない。

### 結果の読み取り

- 10エポックと20エポックの学習後format_rateの差：{(b['after']['format_rate']-a['after']['format_rate'])*100:+.2f}ポイント。
- 学習後content_rateの差：{(b['after']['content_rate']-a['after']['content_rate'])*100:+.2f}ポイント。
- 学習時間の比（10/20エポック）：{b['training_seconds']/a['training_seconds']:.3f}。
- 評価は21問、各条件1回である。GPUの計算や実行環境の影響もあるため、学習回数と性能の一般的な関係までは断定しない。エポック数の変更に伴い、cosineスケジュールの総ステップ数に対するwarmupの比率も変わる。

### ファインチューニングが向く場面・向かない場面

JSON形式、必須キー、定型文や表現を繰り返し学習させたい場面に向く。形式遵守率と内容正解率は別の指標であり、JSONが正しくても根拠のない断定は起こり得る。頻繁に更新される事実の確認や、学習データにない事項の正確な回答を学習だけに任せる用途には注意が必要である。最新情報は検索等で根拠を取得し、確認が必要な質問への対応も別途評価する。

## 実行と検証

- 条件別のresult.jsonと評価出力をresults/に保存した。
- 集計値を21問の評価明細から再計算し、比較する設定差がnum_train_epochsだけであること、ライブラリ版と評価質問・正解が同じであることを確認した。
- AIをノートブックの計測・保存処理と提出文の整理に利用した。実測値は保存した実行結果に基づく。
'''
    (args.out / 'answer.md').write_text(text, encoding='utf-8')
    q3 = '基礎課題として、学習回数を20エポックから10エポックに変更して比較しました。モデル・評価質問・seed・生成条件をそろえ、形式遵守率と内容正解率を分けて評価した点を工夫しました。最終lossと学習時間も保存し、結果を再確認できるようにしました。学習済みadapterが残ると公平な比較にならない点に注意が必要でした。回答形式が整うことと内容の正確さは別だと捉え、学習量・性能・時間の関係を実測値に基づいて考察した点を評価していただきたいです。'
    assert len(q3) <= 400
    (args.out.parent.parent / 'submission_form.txt').write_text('Q1: 基礎｜Basic\n\nQ2: https://github.com/kane-suke/lecture-ai-engineering-2026\n\nQ3:\n' + q3 + f'\n\nQ3文字数: {len(q3)}\n', encoding='utf-8')
    print('\n'.join(table))
    print('Created:', args.out / 'answer.md')


if __name__ == '__main__':
    main()
