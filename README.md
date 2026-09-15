# SS-RAG — Semantic Super RAG

نسخة عربية مبسطة ومستقلة مستوحاة من [Hyper-RAG](https://github.com/iMoonLab/Hyper-RAG):
نظام **Retrieval-Augmented Generation يقوده Hypergraph** يلتقط العلاقات الثنائية (low-order)
والعلاقات متعددة الأطراف (high-order / hyperedges) لتقليل الهلوسة.

## الفكرة (مثل Hyper-RAG)
1. **بناء المعرفة**: تقطيع النصوص → استخراج كيانات + علاقات ثنائية + علاقات عالية (hyperedges تضم 3+ كيانات) → تخزين Hypergraph + ثلاث قواعد فيكتور (entities / relations / chunks).
2. **الاستعلام**: استرجاع كيانات أولية → **انتشار طبقة واحدة** على الـhypergraph → تجميع السياق → توليد الإجابة بـLLM.
3. **الأوضاع**: `hyper` (كامل) / `hyper-lite` (كيانات+مقاطع، أسرع ×2) / `graph` (ثنائية فقط) / `naive` (مقاطع فقط) / `llm` (بدون استرجاع).

## يعمل بدون مفاتيح
- بدون `my_config.py` يعمل **offline**: استخراج regex عربي/إنجليزي + تضمين hashing محلي + إجابات استخراجية.
- مع مفاتيح OpenAI-compatible: استخراج LLM + تضمين سحابي + إجابات توليدية.

## التركيب
```
SS-RAG/
  ssrag/           # النواة: base, utils, llm, prompt, storage, operate, ssrag.py
  examples/ssrag_demo.py + mock_data.txt
  reproduce/Step_0..3  # مثل Hyper-RAG
  evaluate/            # selection + scoring
  service_api.py       # FastAPI + streaming
  testHTML_light.html
```

## التثبيت
```bash
cd SS-RAG
pip install -r requirements.txt
cp config_temp.py my_config.py   # اختياري: ضع مفاتيحك
```

## تشغيل سريع (offline)
```bash
python examples/ssrag_demo.py
# أو خطوة بخطوة:
python reproduce/Step_0.py mock
python reproduce/Step_1.py mock
python reproduce/Step_2_extract_question.py mock
python reproduce/Step_3_response_question.py mock hyper
python reproduce/Step_3_response_question.py mock hyper-lite
python evaluate/evaluate_by_scoring.py mock hyper
```

## API
```bash
uvicorn service_api:app --port 8000
# POST /query  {"question":"...","mode":"hyper"}
# POST /query_stream  (token streaming)
# GET /healthz
```
ثم افتح `testHTML_light.html` في المتصفح.

## النشر على Vercel
المستودع جاهز: `pyproject.toml` يحدد `[tool.vercel] entrypoint = "service_api:app"`
والتثبيت يتم من `requirements.txt` تلقائياً. اضبط متغيرات البيئة في Vercel:
`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` و`EMB_*` (أو اتركها فارغة للوضع offline)،
واختيارياً `SSRAG_DATA_NAME` و`SSRAG_MODE`. ملاحظة: على Vercel يُستخدم `/tmp`
كمجلد عمل (نظام الملفات للقراءة فقط)، والفهرس المبني محلياً في `caches/` غير
منشور (مستبعد في `.gitignore`) — أي أن Retrieval سيعمل على فهرس فارغ ما لم
تُحمّل بياناتك وتبني الفهرس بطريقة أخرى.

## الفرق عن Hyper-RAG الأصلي
| | Hyper-RAG | SS-RAG |
|---|---|---|
| Hypergraph DB | مكتبة `hypergraph-db` خارجية | تخزين JSON ذاتي (`HypergraphStorage`) |
| Vector DB | `nano-vectordb` | `NumpyVectorStorage` ذاتي (cosine brute-force) |
| يعمل offline | لا (يحتاج LLM) | نعم (regex + hashing embeddings) |
| عربي | غير مدعوم رسمياً | مدعوم (prompts + fallback عربي) |
| streaming | نعم | نعم (`/query_stream` + `astream_query`) |

رخصة Apache-2.0.
