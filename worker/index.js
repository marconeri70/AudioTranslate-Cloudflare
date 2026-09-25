import { Buffer } from "node:buffer";

const WHISPER_MODEL="@cf/openai/whisper-large-v3-turbo";
const FILTER_MODEL="@cf/meta/llama-3.3-70b-instruct-fp8-fast";
const TRANSLATE_MODEL="@cf/meta/m2m100-1.2b";
const jsonHeaders={"content-type":"application/json; charset=utf-8"};

function json(data,status=200){
  return new Response(JSON.stringify(data),{status,headers:jsonHeaders});
}
function secondsFromVttTime(value){
  const p=value.trim().split(":").map(Number);
  if(p.length===3)return p[0]*3600+p[1]*60+p[2];
  if(p.length===2)return p[0]*60+p[1];
  return Number(value)||0;
}
function parseVtt(vtt=""){
  const lines=vtt.replace(/\r/g,"").split("\n"),out=[];
  for(let i=0;i<lines.length;i++){
    const line=lines[i].trim();
    if(!line.includes("-->"))continue;
    const[a,b]=line.split("-->").map(x=>x.trim()),text=[];
    for(let j=i+1;j<lines.length&&lines[j].trim();j++)text.push(lines[j].trim());
    if(text.length)out.push({start:secondsFromVttTime(a),end:secondsFromVttTime(b),text:text.join(" ")});
  }
  return out;
}
function normalizeAsrSegments(asr,durationHint){
  let raw=[];
  if(Array.isArray(asr?.segments))raw=asr.segments;
  else if(Array.isArray(asr?.transcription_info))raw=asr.transcription_info;
  else if(asr?.vtt)raw=parseVtt(asr.vtt);
  const normalized=raw.map((s,i)=>({
    index:i,
    start:Number(s.start??s.start_time??0)||0,
    end:Number(s.end??s.end_time??0)||0,
    text:String(s.text??s.transcript??"").replace(/\s+/g," ").trim()
  })).filter(s=>s.text);
  if(!normalized.length&&String(asr?.text||"").trim()){
    normalized.push({index:0,start:0,end:Number(durationHint)||0,text:String(asr.text).replace(/\s+/g," ").trim()});
  }
  return normalized;
}
function extractJsonResponse(output){
  const value=output?.response??output?.result??output;
  if(value&&typeof value==="object")return value;
  if(typeof value!=="string")throw new Error("Risposta strutturata non valida dal filtro lingua.");
  try{return JSON.parse(value)}
  catch{
    const first=value.indexOf("{"),last=value.lastIndexOf("}");
    if(first>=0&&last>first)return JSON.parse(value.slice(first,last+1));
    throw new Error("Il filtro lingua non ha restituito JSON valido.");
  }
}
function extractTranslation(output){
  return String(output?.translated_text??output?.translation??output?.text??output?.response??"").trim();
}

const EN_CUES=new Set([
  "the","this","that","these","those","if","you","your","we","our","they","it","is","are","was","were",
  "have","has","had","and","or","but","with","from","into","for","to","of","on","at","by","as","can",
  "could","would","should","will","make","generate","demonstration","metal","field","frequency","radiation",
  "electron","electronic","energy","photovoltaic","spectrum","spectroscopy","model","structure","radio",
  "magnetic","electric","wave","wavelength","oscillation","plane","here","there","put","consume"
]);
const IT_CUES=new Set([
  "perché","perche","sono","questo","questa","questi","queste","quello","quella","dentro","allora","quindi",
  "abbiamo","avete","hanno","anche","della","delle","degli","nella","nelle","sulla","sulle","cosa","come",
  "bambini","albero","elettronico","elettronica","potrebbero","utilizzare","diversi","diverse","qui","lì",
  "cioè","cioe","oppure","invece","grazie","ragazzi","professore","professoressa"
]);
function words(text){
  return String(text||"").toLowerCase().match(/[a-zà-ÿ']+/g)||[];
}
function cueScores(text){
  const w=words(text);
  let en=0,it=0;
  for(const token of w){
    if(EN_CUES.has(token))en++;
    if(IT_CUES.has(token))it++;
  }
  if(/[àèéìòù]/i.test(text))it+=2;
  if(/\b(in the|if you|this is|that is|we have|you can|we can|make this|generate a|put the|of the|to the)\b/i.test(text))en+=3;
  if(/\b(perché|che cosa|ci sono|è questo|sono diversi|dentro|questo è|questa è)\b/i.test(text))it+=3;
  return {en,it,count:w.length};
}
function clearlyEnglish(text){
  const s=cueScores(text);
  return s.en>=2 && s.it===0;
}
function clearlyItalian(text){
  const s=cueScores(text);
  return s.it>=2 && s.it>s.en;
}
function surroundingEnglish(prev,next){
  return clearlyEnglish(prev)||clearlyEnglish(next);
}
function repairClassification(raw,c,chunkLanguage,prev,next){
  const text=String(raw?.text||"").trim();
  const lang=String(c?.language||"uncertain");
  const scores=cueScores(text);
  const chunkEn=/^en/i.test(String(chunkLanguage||""));

  if(clearlyItalian(text))return c;

  if(clearlyEnglish(text)){
    return {...c,language:"en",confidence:scores.en>=3?"high":"medium",english_text:text,excluded_text:""};
  }

  if(chunkEn && scores.count<=8 && scores.it===0 && (scores.en>=1 || surroundingEnglish(prev,next))){
    return {...c,language:"en",confidence:scores.en>=1?"medium":"low",english_text:text,excluded_text:""};
  }

  if(lang==="it" && scores.count<=8 && scores.it===0){
    return {...c,language:"uncertain",confidence:"low",english_text:"",excluded_text:text};
  }

  return c;
}

async function classifySegments(env,segments,filterMode,context){
  const strict=filterMode==="strict";
  const schema={
    type:"object",
    properties:{
      segments:{
        type:"array",
        items:{
          type:"object",
          properties:{
            index:{type:"integer"},
            language:{type:"string",enum:["en","it","mixed","other","noise","uncertain"]},
            confidence:{type:"string",enum:["high","medium","low"]},
            english_text:{type:"string"},
            excluded_text:{type:"string"}
          },
          required:["index","language","confidence","english_text","excluded_text"],
          additionalProperties:false
        }
      }
    },
    required:["segments"],
    additionalProperties:false
  };

  const payload=segments.map((s,i)=>({
    index:s.index,
    previous:i>0?segments[i-1].text:"",
    text:s.text,
    next:i<segments.length-1?segments[i+1].text:""
  }));

  const system=`Sei un filtro linguistico per una lezione universitaria prevalentemente in inglese.
In aula possono esserci brevi conversazioni in italiano, rumori e frasi miste.
Materia/contesto: ${context||"lezione universitaria"}.

REGOLA FONDAMENTALE:
- una frase grammaticalmente inglese va classificata EN anche se è molto breve;
- la brevità NON è un motivo per usare uncertain;
- esempi come "if you put the metal here", "In the demonstration", "Generate a photovoltaic", "make this" sono INGLESE;
- termini scientifici isolati come "radio", "frequency", "field", "electron", "energy" vanno letti nel contesto precedente e successivo;
- usa IT solo quando ci sono vere parole o strutture italiane;
- usa uncertain solo per parlato realmente incomprensibile, troncato o linguisticamente ambiguo.

Per OGNI segmento:
- classifica en, it, mixed, other, noise oppure uncertain;
- english_text deve contenere SOLO parole realmente pronunciate in inglese;
- NON tradurre italiano in inglese;
- excluded_text contiene la parte non inglese o davvero dubbia;
- per mixed separa inglese e italiano soltanto se chiaramente distinguibili;
- preserva terminologia scientifica, formule e nomi propri;
- non riassumere e non parafrasare;
- usa previous e next SOLO per capire la lingua del segmento centrale.

${strict
  ?"MODALITÀ FORTE: escludi l'italiano con decisione, ma NON trasformare brevi frasi inglesi corrette in uncertain."
  :"MODALITÀ BILANCIATA: conserva l'inglese chiaramente prevalente anche nei segmenti misti."}`;

  const out=await env.AI.run(FILTER_MODEL,{
    messages:[
      {role:"system",content:system},
      {role:"user",content:JSON.stringify(payload)}
    ],
    temperature:0,
    max_tokens:2200,
    response_format:{type:"json_schema",json_schema:schema}
  });

  const parsed=extractJsonResponse(out);
  if(!Array.isArray(parsed?.segments))throw new Error("Filtro lingua: array segmenti mancante.");
  return parsed.segments;
}

async function translateEnglish(env,text){
  const clean=String(text||"").trim();
  if(!clean)return"";
  const out=await env.AI.run(TRANSLATE_MODEL,{text:clean,source_lang:"en",target_lang:"it"});
  return extractTranslation(out);
}

async function processChunk(request,env){
  const contentType=request.headers.get("content-type")||"audio/mp4";
  const offset=Number(request.headers.get("x-chunk-offset")||0)||0;
  const durationHint=Number(request.headers.get("x-chunk-duration")||60)||60;
  const discardBefore=Number(request.headers.get("x-discard-before")||0)||0;
  const filterMode=request.headers.get("x-filter-mode")==="balanced"?"balanced":"strict";
  const context=decodeURIComponent(request.headers.get("x-context")||"Physical Chemistry").slice(0,300);

  const bytes=await request.arrayBuffer();
  if(!bytes.byteLength)return json({error:"Spezzone audio vuoto."},400);
  if(bytes.byteLength>12*1024*1024)return json({error:"Spezzone troppo grande: massimo 12 MB."},413);

  const base64=Buffer.from(bytes).toString("base64");
  const asr=await env.AI.run(WHISPER_MODEL,{
    audio:base64,
    task:"transcribe",
    vad_filter:true,
    initial_prompt:`University lecture in English. Subject: ${context}. Preserve scientific and technical terminology exactly. Italian classroom speech may occur and must not be converted into English.`,
    beam_size:5,
    condition_on_previous_text:false,
    no_speech_threshold:.6,
    compression_ratio_threshold:2.4,
    log_prob_threshold:-1,
    hallucination_silence_threshold:2
  });

  const rawSegments=normalizeAsrSegments(asr,durationHint)
    .filter(s=>s.end>discardBefore||s.start>=discardBefore);

  const detectedLanguage=asr?.transcription_info?.language??asr?.language??null;

  if(!rawSegments.length){
    return json({englishText:"",italianText:"",englishSegments:[],excludedSegments:[],uncertainSegments:[],rawText:String(asr?.text||""),detectedLanguage});
  }

  const classified=await classifySegments(env,rawSegments,filterMode,context);
  const byIndex=new Map(classified.map(x=>[Number(x.index),x]));
  const englishSegments=[],excludedSegments=[],uncertainSegments=[];

  for(let i=0;i<rawSegments.length;i++){
    const raw=rawSegments[i];
    let c=byIndex.get(raw.index);
    if(!c)continue;

    c=repairClassification(
      raw,c,detectedLanguage,
      i>0?rawSegments[i-1].text:"",
      i<rawSegments.length-1?rawSegments[i+1].text:""
    );

    const start=offset+raw.start,end=offset+raw.end;
    const english=String(c.english_text||"").replace(/\s+/g," ").trim();
    const excluded=String(c.excluded_text||"").replace(/\s+/g," ").trim();
    const language=c.language||"uncertain";
    const confidence=c.confidence||"low";

    if(english)englishSegments.push({start,end,text:english,language,confidence});
    if(excluded){
      const item={start,end,text:excluded,language,confidence};
      if(language==="it"||language==="mixed")excludedSegments.push(item);
      else uncertainSegments.push(item);
    }
  }

  const englishText=englishSegments.map(x=>x.text).join(" ").trim();
  let italianText="";
  if(englishText){
    try{italianText=await translateEnglish(env,englishText)}catch{italianText=""}
  }

  return json({
    englishText,italianText,englishSegments,excludedSegments,uncertainSegments,
    rawText:String(asr?.text||""),detectedLanguage,contentType
  });
}

export default{
  async fetch(request,env){
    const url=new URL(request.url);
    if(url.pathname==="/api/health"&&request.method==="GET"){
      return json({
        ok:true,version:"6.3.0",provider:"Cloudflare Workers AI",
        whisper:WHISPER_MODEL,filter:FILTER_MODEL,translation:TRANSLATE_MODEL,storage:false
      });
    }
    if(url.pathname==="/api/process-chunk"&&request.method==="POST"){
      try{return await processChunk(request,env)}
      catch(error){
        console.error(error);
        return json({error:error instanceof Error?error.message:String(error)},500);
      }
    }
    if(url.pathname.startsWith("/api/"))return json({error:"Endpoint non trovato."},404);
    return new Response("Not Found",{status:404});
  }
};
