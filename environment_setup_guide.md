# 仮想環境セットアップガイド (Environment Setup Guide)

本リポジトリ（`affective_empathy_eval`）の実験・テスト・分析を実行するための仮想環境構築手順です。
AGENTS.md の規定に従い、グローバル Python 環境ではなく、必ずプロジェクト専用の仮想環境（`.venv`）を構築して作業してください。

---

## 1. 自動セットアップ（推奨: 1コマンド）

リポジトリルートに用意された自動構築スクリプトを実行します：

```bash
./setup_env.sh
```

- システム内に `uv` がインストールされている場合は `uv` を用いて数秒で構築します。
- `uv` がない場合は標準の `python3 -m venv` を用いて構築し、`pip` をアップグレードした上で全依存パッケージをインストールします。
- 構築完了後、主要パッケージ（`torch`, `transformers`, `POT`, `scipy`, `sklearn` 等）のインポートテストが自動実行されます。

---

## 2. 手動セットアップ手順

### 方法 A: `uv` を使用する場合（推奨・最速）

```bash
# 1. 仮想環境の作成 (Python 3.12)
uv venv .venv --python python3.12

# 2. 仮想環境の有効化
source .venv/bin/activate

# 3. 開発用パッケージを含めてインストール (editable モード)
uv pip install -e ".[dev]"
```

### 方法 B: 標準の `venv` を使用する場合

```bash
# 1. 仮想環境の作成
python3.12 -m venv .venv

# 2. 仮想環境の有効化
source .venv/bin/activate

# 3. pip の更新とパッケージインストール
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

---

## 3. 構成ファイル一覧

| ファイル | 役割 |
|---|---|
| [`setup_env.sh`](file:///mnt/nas/home/hiromi/src/emo/setup_env.sh) | 仮想環境自動構築・検証シェルスクリプト |
| [`pyproject.toml`](file:///mnt/nas/home/hiromi/src/emo/pyproject.toml) | プロジェクト仕様・全依存関係（POT含む）・パッケージ定義 |
| [`requirements.txt`](file:///mnt/nas/home/hiromi/src/emo/requirements.txt) | pip 用依存関係一覧ファイル |
| [`.python-version`](file:///mnt/nas/home/hiromi/src/emo/.python-version) | 指定 Python バージョン（`3.12`）固定ファイル |
| [`pytest.ini`](file:///mnt/nas/home/hiromi/src/emo/pytest.ini) | pytest 設定（テスト探索パス・PYTHONPATH設定） |

---

## 4. 動作確認

仮想環境が正しくセットアップされたか確認するために、ルートからテストを実行します：

```bash
.venv/bin/pytest -q
```
