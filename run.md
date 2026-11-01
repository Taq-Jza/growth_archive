# 本地运行

## 1. 装依赖

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows
pip install -r requirements.txt
```

## 2. 配置与验证

```bash
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY
python scripts/check_llm.py
```

`check_llm.py` 会打印非流式耗时与流式首字延迟。首字超过 3 秒就换小一档的模型。
**没有 key 也能启动**，问答会降级为「返回检索到的原文片段」，其余功能不受影响。

## 3. 启动

```bash
uvicorn app.main:app --reload --port 8000
```

| 路径 | 说明 |
|---|---|
| `/` | 概览：一句话记录、能力分布饼图、要点卡、记录列表 |
| `/c/{kind}` | 分类页，kind 取 course / skill / award / project / trait |
| `/cards` | 全部要点卡（可按 kind 筛选、分页） |
| `/?zoom=overview` 或 `/?zoom=detail&idx=N` | 直接打开能力分布的放大视图 |
| `/course/question` | 课程助手：问题分析 |
| `/new` | 添加资料（上传文件 / 拍照 / 粘贴文本） |
| `/search` | 检索，支持 `q`、`kind`、`frm`、`to`、`page` |
| `/doc/{id}` | 记录详情；带 `?cid=` 定位并高亮被引用的片段 |
| `/doc/{id}/edit` | 编辑记录（会重建索引与向量） |
| `/doc/{id}/interview` | 面试追问预演 |
| `/ask?q=` | 带引用问答，可用 `&scope=course` 限定分类 |
| `/api/ask?q=` | SSE 接口，供前端使用 |
| `/upload` | 文件上传导入（multipart） |
| `/match` | 岗位匹配 |
| `/register` | 用邮箱自建账号（口令至少 6 位） |
| `/login` | 邮箱 + 口令登录 |
| `/demo` | 一键体验登录（不需要口令） |
| `/healthz` | 健康检查 |

打开 http://127.0.0.1:8000，两种进法：

- **自己用**：点「还没有账号？用邮箱创建一个」，填邮箱和口令（≥ 6 位）即可。
  新账号是空档案，首页给「载入示例档案」可以一键填入 26 条示例。
- **直接看效果**：点「或者直接看体验账号」，不用口令，进去就是已有数据的那份档案。

体验账号的邮箱与口令取自 `.env` 的 `DEMO_EMAIL` / `DEMO_PASSWORD`，首次启动自动创建。这个版本没有找回口令，忘了只能换邮箱重建账号。

## 4. 目录

```
app/
  main.py       路由
  db.py         SQLite schema 与查询
  auth.py       PBKDF2 + 签名 Cookie
  textproc.py   jieba 分词、切块、高亮
  ingest.py     入库与建索引
  search.py     FTS5 检索 + RRF 融合
  seed.py       示例库载入
  templates/    服务端模板
data/
  seed.json     示例数据
  app.db        运行时生成，已 gitignore
```

## 5. 为什么用 SQLite 而不是 PostgreSQL

当前规模是单用户、数十条记录、千级切片：

- SQLite 零运维，ECS 上不用装数据库，省掉约 5 小时环境调试；
- 200 条向量的余弦相似度在 Python 里暴力算是毫秒级；
- FTS5 + jieba 预分词足以支撑中文检索。

代价是并发写入能力弱；单用户场景下可接受。

**迁移路径**：`db.py` 的 SQL 只用了标准语法，向量字段存的是 JSON 文本。
真要迁到 PostgreSQL，把 `chunks.embedding` 换成 `vector(1024)`、FTS5 换成 `tsvector` 即可，
上层服务代码不用动。

## 6. 数据重置

```bash
rm -f data/app.db data/app.db-wal data/app.db-shm
```
重启后会重新建表并创建体验账号。

## 6.5 向量与混合检索

入库时会自动为切片生成向量（bge-m3，1024 维），存在 `chunks.embedding` 里。
检索时关键词路与语义路各召回 20 条，RRF 融合后交给模型重排。

- 生成失败**不影响入库**：文本照样能写进库、能按关键词搜到，只是暂时不参与语义检索。
- 补齐缺失向量：

```bash
.venv/Scripts/python.exe scripts/embed_backfill.py            # 只补缺失的
.venv/Scripts/python.exe scripts/embed_backfill.py --rebuild  # 换模型后全量重算
```

- 语义路不可用时自动退回纯关键词，`/ask` 页顶部会显示当前走的是哪条路。
- 26 条示例记录 + 生成向量约 5 秒。

## 7. 部署到服务器

最简单的方式是用仓库里的脚本：

```bash
bash scripts/deploy.sh     # 装依赖 → 检查 .env → 重启 → 健康检查
```

手工方式：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
nohup .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 >> logs/app.log 2>&1 &
```

安全组放行 8000 端口，访问 `http://<公网IP>:8000`。不需要域名。

## 8. 进程托管与备份

用 systemd 托管（开机自启、崩溃自动拉起）：

```bash
sudo cp deploy/growth-archive.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now growth-archive
sudo systemctl status growth-archive
```

unit 里的路径按实际部署目录改（默认 `/opt/growth-archive`）。

备份：

```bash
bash scripts/backup.sh     # 数据库快照 + 静态数据打包，保留最近 7 份
```

建议加定时任务：

```cron
20 3 * * * cd /opt/growth-archive && bash scripts/backup.sh >> logs/backup.log 2>&1
```
