from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime, timezone
import os, json, re, time

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / '.env')
DATA = json.loads((BASE / 'data' / 'businesses.json').read_text(encoding='utf-8'))
FEEDBACK_FILE = BASE / 'data' / 'feedback.json'

app = FastAPI(title='NAmma BiZZ AI API', version='1.0.0')
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_REQUESTS = 60
_rate_limit_log = {}


@app.middleware('http')
async def api_guard(request: Request, call_next):
    """Add basic browser protections and a small abuse guard for the demo API."""
    if request.url.path.startswith('/api/'):
        now = time.monotonic()
        client = request.client.host if request.client else 'unknown'
        recent = [stamp for stamp in _rate_limit_log.get(client, [])
                  if now - stamp < RATE_LIMIT_WINDOW_SECONDS]
        if len(recent) >= RATE_LIMIT_REQUESTS:
            return JSONResponse(status_code=429, content={'detail': 'Too many requests. Please try again shortly.'})
        recent.append(now)
        _rate_limit_log[client] = recent
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

class MatchRequest(BaseModel):
    requirement: str = Field(min_length=2, max_length=500)
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    accuracy: float | None = Field(None, ge=0, le=100000)

class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)

class EstimateRequest(BaseModel):
    product: str = Field(min_length=2, max_length=200)
    quantity: float = Field(gt=0, le=1_000_000)
    budget: float = Field(gt=0, le=100_000_000)
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)

class FeedbackRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    business_id: int | None = Field(None, alias='businessId')
    reasons: list[str] = Field(default_factory=list, max_length=5)
    comment: str = Field(default='', max_length=1000)

DOMAINS = [b['category'] for b in DATA if b.get('category')]
DOMAINS = list(dict.fromkeys(DOMAINS))

SCHEMA = {
  'type':'object','additionalProperties':False,
  'properties':{
    'domain':{'type':'string'},
    'product_terms':{'type':'array','items':{'type':'string'}},
    'quantity':{'type':'string'},
    'location':{'type':'string'},
    'constraints':{'type':'array','items':{'type':'string'}},
    'business_ids':{'type':'array','items':{'type':'integer'}},
    'reasons':{'type':'array','items':{'type':'string'}},
    'summary':{'type':'string'}
  },
  'required':['domain','product_terms','quantity','location','constraints','business_ids','reasons','summary']
}

def catalog_text():
    rows=[]
    for b in DATA:
        products=[]
        for p in b.get('productsDetailed',[]):
            products.append(f"{p['name']} | Price {p['price']} | {p['moq']} | Units sold {p.get('unitsSold', 0)} | {p['description']}")
        rows.append(f"ID {b['id']} | {b['name']} | {b['category']} | {b['location']} | Coordinates {b.get('latitude')},{b.get('longitude')} | {b['description']} | Products: {' ; '.join(products)}")
    return '\n'.join(rows)

def distance_km(latitude, longitude, business):
    from math import atan2, cos, radians, sin, sqrt
    earth_radius = 6371.0
    d_lat = radians(float(business['latitude']) - latitude)
    d_lon = radians(float(business['longitude']) - longitude)
    lat1 = radians(latitude)
    lat2 = radians(float(business['latitude']))
    a = sin(d_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(d_lon / 2) ** 2
    return earth_radius * 2 * atan2(sqrt(a), sqrt(1 - a))

def price_number(value):
    match = re.search(r'(\d[\d,]*(?:\.\d+)?)', str(value))
    return float(match.group(1).replace(',', '')) if match else None

def product_matches(product_query, product_name):
    query_words = set(re.findall(r'[a-z0-9]+', product_query.lower()))
    name_words = set(re.findall(r'[a-z0-9]+', product_name.lower()))
    ignored = {'need', 'want', 'for', 'the', 'and', 'service', 'work'}
    return bool((query_words - ignored) & name_words)


def approximate_location(latitude, longitude):
    """Round GPS before it is sent to the external AI provider."""
    if latitude is None or longitude is None:
        return None
    return round(latitude, 3), round(longitude, 3)


def distance_label(km):
    return f'{round(km * 1000)} m away' if km < 1 else f'{km:.1f} km away'


def fallback_match(requirement, latitude=None, longitude=None):
    """Deterministic catalog matching used when AI is unavailable or inconclusive."""
    query = re.findall(r'[a-z0-9]+', requirement.lower())
    ignored = {'need', 'want', 'for', 'the', 'and', 'from', 'in', 'of', 'with', 'wholesale', 'supplier'}
    tokens = {token for token in query if token not in ignored and len(token) > 2}
    scored = []
    for business in DATA:
        profile = ' '.join([
            business.get('category', ''), business.get('name', ''), business.get('description', ''),
            *business.get('products', []), *business.get('tags', []),
            *(product.get('name', '') for product in business.get('productsDetailed', [])),
        ]).lower()
        hits = sorted({token for token in tokens if token in profile})
        score = min(95, len(hits) * 16)
        reasons = []
        if hits:
            reasons.append('Catalog keywords match: ' + ', '.join(hits[:3]))
        if business.get('category', '').lower() in requirement.lower():
            score += 25
            reasons.insert(0, 'Business domain matches')
        if not score:
            continue
        item = dict(business)
        item['_score'] = min(99, score)
        item['_reasons'] = reasons or ['General catalog relevance']
        if latitude is not None and longitude is not None:
            km = distance_km(latitude, longitude, business)
            item['_distance'] = km
            item['distanceKm'] = round(km, 2)
            item['distanceText'] = distance_label(km)
            item['_reasons'].append(item['distanceText'] + ' from you')
        scored.append(item)
    scored.sort(key=lambda item: (-item['_score'], item.get('_distance', float('inf'))))
    results = []
    for item in scored[:8]:
        item['match'] = f"{item.pop('_score')}%"
        item['reasons'] = item.pop('_reasons')[:4]
        item.pop('_distance', None)
        results.append(item)
    return results


def business_reasons(business, requirement, parsed_reasons=None):
    reasons = []
    q = requirement.lower()
    if business.get('category', '').lower() in q:
        reasons.append('Business domain matches')
    if any(product_matches(requirement, product.get('name', '')) for product in business.get('productsDetailed', [])):
        reasons.append('Requested product appears in catalog')
    if 'wholesale' in q or 'bulk' in q:
        reasons.append('Wholesale or bulk requirement')
    if parsed_reasons and not reasons:
        reasons.append(parsed_reasons[0])
    return reasons or ['Selected from the supplied catalog']


def append_feedback(payload):
    records = []
    if FEEDBACK_FILE.exists():
        try:
            records = json.loads(FEEDBACK_FILE.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            records = []
    records.append(payload)
    FEEDBACK_FILE.write_text(json.dumps(records[-5000:], ensure_ascii=False, indent=2), encoding='utf-8')

CATALOG = catalog_text()

def match_catalog_text():
    rows=[]
    for business in DATA:
        products=' ; '.join(
            f"{product['name']} | {product['moq']}"
            for product in business.get('productsDetailed', [])
        )
        keywords=', '.join(business.get('products', []) + business.get('tags', []))
        rows.append(f"ID {business['id']} | {business['name']} | {business['category']} | {business['location']} | Keywords {keywords} | {business.get('description', '')} | {products}")
    return '\n'.join(rows)

MATCH_CATALOG = match_catalog_text()

def qa_catalog_text():
    rows=[]
    for business in DATA:
        products=' ; '.join(
            f"{product['name']} | {product['price']} | {product['moq']} | sold {product.get('unitsSold', 0)}"
            for product in business.get('productsDetailed', [])
        )
        rows.append(f"ID {business['id']} | {business['name']} | {business['category']} | {business['location']} | {products}")
    return '\n'.join(rows)

QA_CATALOG = qa_catalog_text()

@app.get('/api/health')
def health():
    return {
        'status': 'ok',
        'businesses': len(DATA),
        'products': sum(len(b.get('productsDetailed', [])) for b in DATA),
        'ai_configured': bool(os.getenv('GROQ_API_KEY')),
        'fallback_matching': True,
        'feedback_storage': 'server-json',
    }

@app.post('/api/ai-match')
def ai_match(req: MatchRequest):
    if not req.requirement.strip():
        raise HTTPException(400,'Requirement is required')
    key=os.getenv('GROQ_API_KEY')
    if not key:
        results = fallback_match(req.requirement, req.latitude, req.longitude)
        return {
            'detected': {'domain': 'Catalog fallback', 'location': 'Madurai'},
            'requestedProducts': [],
            'quantity': 'Not specified',
            'constraints': [],
            'summary': 'AI is unavailable, so results were matched using the local catalog.',
            'results': results,
            'noMatch': not results,
            'aiModel': 'catalog-rule-matcher',
        }
    model=os.getenv('GROQ_MODEL','openai/gpt-oss-20b')
    instructions=(
      'You are the NAmma BiZZ wholesale matching AI for Madurai. '
      'Only recommend businesses that exist in the supplied catalog. Never invent a business, product, price, MOQ, rating, address, or capability. '
      'All businesses are wholesale suppliers in the Madurai demo catalog. Interpret natural language, synonyms, spelling variations, quantity, material, use case and constraints. '
    'Return at most 8 business IDs, ordered by product relevance and customer distance. When products are similarly relevant, prefer the closer business. If nothing is genuinely relevant, return an empty business_ids array. '
    'The domain must be one of: '+', '.join(DOMAINS)+'. Return only valid JSON matching the requested response structure.'
    )
    location_context = ''
    if req.latitude is not None and req.longitude is not None:
        safe_latitude, safe_longitude = approximate_location(req.latitude, req.longitude)
        location_context = (
            f"\nCUSTOMER APPROXIMATE LOCATION: latitude {safe_latitude}, longitude {safe_longitude}. "
            f"Prefer relevant businesses closer to the customer. Use the precomputed distances below as a ranking factor.\n"
            + '\n'.join(
                f"SHOP DISTANCE: ID {business['id']} = {distance_km(safe_latitude, safe_longitude, business):.2f} km"
                for business in DATA
            )
        )
    user=(f"CUSTOMER REQUIREMENT:\n{req.requirement}{location_context}\n\nCOMPACT CATALOG:\n{MATCH_CATALOG}")
    try:
                response=Groq(api_key=key).chat.completions.create(
          model=model,
                    messages=[
                        {'role':'system','content':instructions},
                        {'role':'user','content':user},
                    ],
                    response_format={'type':'json_object'},
                    temperature=0.1,
        )
                parsed=json.loads(response.choices[0].message.content)
    except Exception:
        results = fallback_match(req.requirement, req.latitude, req.longitude)
        return {
            'detected': {'domain': 'Catalog fallback', 'location': 'Madurai'},
            'requestedProducts': [],
            'quantity': 'Not specified',
            'constraints': [],
            'summary': 'The AI provider was unavailable, so results were matched using the local catalog.',
            'results': results,
            'noMatch': not results,
            'aiModel': 'catalog-rule-matcher',
        }

    allowed={b['id']:b for b in DATA}
    results=[]
    for rank,bid in enumerate(parsed.get('business_ids',[])[:8]):
        if bid not in allowed: continue
        b=dict(allowed[bid])
        # Ranking is model-provided ordering; score is presentation-only, not a factual probability.
        b['match']=str(max(60,100-rank*6))+'%'
        b['reasons']=business_reasons(b, req.requirement, parsed.get('reasons') or [])
        if req.latitude is not None and req.longitude is not None:
            km = distance_km(req.latitude, req.longitude, b)
            b['distanceKm']=round(km, 2)
            b['distanceText']=f'{round(km * 1000)} m away' if km < 1 else f'{km:.1f} km away'
            b['reasons'] = b['reasons'][:3] + [f"{b['distanceText']} from you"]
        results.append(b)
    return {
      'detected':{'domain':parsed.get('domain','General'),'location':parsed.get('location','Madurai')},
      'requestedProducts':parsed.get('product_terms',[]),
      'quantity':parsed.get('quantity','Not specified'),
      'constraints':parsed.get('constraints',[]),
      'summary':parsed.get('summary',''),
      'results':results,
      'noMatch':not results,
      'aiModel':model
    }

@app.post('/api/ask')
def ask_catalog(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, 'Question is required')
    key=os.getenv('GROQ_API_KEY')
    if not key:
        return {'answer': 'AI is unavailable. Try the main search; it can still use local catalog matching.', 'aiModel': 'catalog-rule-matcher'}
    model=os.getenv('GROQ_MODEL','openai/gpt-oss-20b')
    location_context=''
    if req.latitude is not None and req.longitude is not None:
        safe_latitude, safe_longitude = approximate_location(req.latitude, req.longitude)
        location_context='\nCustomer approximate location: latitude '+str(safe_latitude)+', longitude '+str(safe_longitude)+'. Use the following shop distances when answering proximity questions:\n'+ '\n'.join(
            f"ID {business['id']} ({business['name']}): {distance_km(safe_latitude, safe_longitude, business):.2f} km"
            for business in DATA
        )
    instructions=(
        'You are the NAmma BiZZ catalog Q&A assistant for wholesale buyers in Madurai. '
        'Answer only from the supplied catalog and GPS distance list. Never invent businesses, products, prices, stock, delivery promises, ratings, or capabilities. '
        'If the catalog does not contain the answer, say that the demo catalog does not provide it. '
        'Be concise and mention the relevant business or product names. Do not expose internal IDs unless useful.'
    )
    prompt=f"QUESTION:\n{req.question}{location_context}\n\nCOMPACT CATALOG:\n{QA_CATALOG}"
    try:
        response=Groq(api_key=key).chat.completions.create(
            model=model,
            messages=[
                {'role':'system','content':instructions},
                {'role':'user','content':prompt},
            ],
            temperature=0.1,
        )
        answer=response.choices[0].message.content.strip()
    except Exception:
        return {'answer': 'The AI provider is unavailable right now. Please use the catalog search or try again shortly.', 'aiModel': 'catalog-rule-matcher'}
    return {'answer':answer,'aiModel':model}


@app.post('/api/feedback')
def save_feedback(req: FeedbackRequest):
    if req.business_id is not None and not any(b['id'] == req.business_id for b in DATA):
        raise HTTPException(400, 'Unknown business')
    if not req.reasons and not req.comment.strip():
        raise HTTPException(400, 'Feedback must include a reason or comment')
    append_feedback({
        'businessId': req.business_id,
        'reasons': [reason.strip()[:120] for reason in req.reasons if reason.strip()],
        'comment': req.comment.strip(),
        'submittedAt': datetime.now(timezone.utc).isoformat(),
    })
    return {'status': 'saved'}

@app.post('/api/estimate')
def estimate_catalog(req: EstimateRequest):
    if not req.product.strip():
        raise HTTPException(400, 'Product or service is required')
    if req.quantity <= 0 or req.budget <= 0:
        raise HTTPException(400, 'Quantity and budget must be greater than zero')
    offers=[]
    for business in DATA:
        for product in business.get('productsDetailed', []):
            if not product_matches(req.product, product['name']):
                continue
            unit_price=price_number(product['price'])
            if unit_price is None:
                continue
            total=unit_price * req.quantity
            offer={
                'businessId':business['id'],
                'businessName':business['name'],
                'category':business['category'],
                'product':product['name'],
                'unitPrice':unit_price,
                'priceLabel':product['price'],
                'moq':product['moq'],
                'quantity':req.quantity,
                'estimatedTotal':round(total, 2),
                'withinBudget':total <= req.budget,
            }
            if req.latitude is not None and req.longitude is not None:
                km=distance_km(req.latitude, req.longitude, business)
                offer['distanceKm']=round(km, 2)
                offer['distanceText']=f'{round(km * 1000)} m away' if km < 1 else f'{km:.1f} km away'
            offers.append(offer)
    offers.sort(key=lambda offer:(not offer['withinBudget'], offer['estimatedTotal'], offer.get('distanceKm', float('inf'))))
    return {
        'product':req.product,
        'quantity':req.quantity,
        'budget':req.budget,
        'currency':'INR',
        'offers':offers[:20],
        'affordableOffers':[offer for offer in offers if offer['withinBudget']][:20],
        'noMatch':not offers,
    }

# Serve the existing single-page frontend after API routes.
app.mount('/', StaticFiles(directory=str(BASE/'frontend'), html=True), name='frontend')
