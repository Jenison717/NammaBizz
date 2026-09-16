from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import os, json, re

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / '.env')
DATA = json.loads((BASE / 'data' / 'businesses.json').read_text(encoding='utf-8'))

app = FastAPI(title='NAmma BiZZ AI API', version='1.0.0')

class MatchRequest(BaseModel):
    requirement: str
    latitude: float | None = None
    longitude: float | None = None
    accuracy: float | None = None

class AskRequest(BaseModel):
    question: str
    latitude: float | None = None
    longitude: float | None = None

class EstimateRequest(BaseModel):
    product: str
    quantity: float
    budget: float
    latitude: float | None = None
    longitude: float | None = None

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
    return {'status':'ok','businesses':len(DATA),'products':sum(len(b.get('productsDetailed',[])) for b in DATA),'ai_configured':bool(os.getenv('GROQ_API_KEY'))}

@app.post('/api/ai-match')
def ai_match(req: MatchRequest):
    if not req.requirement.strip():
        raise HTTPException(400,'Requirement is required')
    key=os.getenv('GROQ_API_KEY')
    if not key:
        raise HTTPException(503,'GROQ_API_KEY is not configured. Add it to .env and restart the server.')
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
        location_context = (
            f"\nCUSTOMER GPS LOCATION: latitude {req.latitude}, longitude {req.longitude}. "
            f"Prefer relevant businesses closer to the customer. Use the precomputed distances below as a ranking factor.\n"
            + '\n'.join(
                f"SHOP DISTANCE: ID {business['id']} = {distance_km(req.latitude, req.longitude, business):.2f} km"
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
    except Exception as e:
        raise HTTPException(502, f'AI request failed: {e}')

    allowed={b['id']:b for b in DATA}
    results=[]
    for rank,bid in enumerate(parsed.get('business_ids',[])[:8]):
        if bid not in allowed: continue
        b=dict(allowed[bid])
        # Ranking is model-provided ordering; score is presentation-only, not a factual probability.
        b['match']=str(max(60,100-rank*6))+'%'
        b['reasons']=(parsed.get('reasons') or [])[:4]
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
        raise HTTPException(503, 'GROQ_API_KEY is not configured. Add it to .env and restart the server.')
    model=os.getenv('GROQ_MODEL','openai/gpt-oss-20b')
    location_context=''
    if req.latitude is not None and req.longitude is not None:
        location_context='\nCustomer GPS: latitude '+str(req.latitude)+', longitude '+str(req.longitude)+'. Use the following exact shop distances when answering proximity questions:\n'+ '\n'.join(
            f"ID {business['id']} ({business['name']}): {distance_km(req.latitude, req.longitude, business):.2f} km"
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
    except Exception as e:
        raise HTTPException(502, f'AI request failed: {e}')
    return {'answer':answer,'aiModel':model}

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
