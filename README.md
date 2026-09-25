# AudioTranslate Professional v6 — GitHub + Cloudflare

Questa versione sostituisce OpenAI API, Render e il motore Whisper locale.

## Architettura
1. Il browser carica il file della lezione.
2. FFmpeg WebAssembly divide localmente il file in M4A validi da 45/60/90 secondi.
3. Ogni spezzone viene inviato a Cloudflare Worker.
4. `@cf/openai/whisper-large-v3-turbo` trascrive senza forzare la lingua.
5. `@cf/meta/llama-3.1-8b-instruct` separa EN / IT / misto / incerto.
6. `@cf/meta/m2m100-1.2b` traduce l'inglese in italiano.
7. Il browser ricompone TXT, SRT, VTT e JSON.
8. Gli spezzoni audio non vengono salvati in R2 o su altri storage.

## Perché FFmpeg nel browser
Tagliare un M4A semplicemente a byte può produrre pezzi non validi. FFmpeg effettua un vero remux e genera piccoli M4A completi. Per gli M4A/AAC compatibili usa `-c:a copy`, quindi è veloce e senza perdita. Solo per formati non compatibili usa una conversione AAC di ripiego.

## Deploy con GitHub + Cloudflare
Carica questa cartella nel repository `AudioTranslate-EN-IT/AudioTranslate` con nome `cloudflare-v6`.

Su Cloudflare:
1. Workers & Pages → Create / Import repository.
2. Collega GitHub e scegli `AudioTranslate-EN-IT/AudioTranslate`.
3. Root directory: `cloudflare-v6`.
4. Build command: `npm run build`.
5. Deploy command: `npx wrangler deploy`.

Cloudflare leggerà `wrangler.jsonc`, userà il binding Workers AI `AI` e pubblicherà frontend + API nello stesso Worker.

Non servono `OPENAI_API_KEY`, Render o chiavi segrete nel browser.

## Impostazioni consigliate per Prima ora physical.m4a
- Materia: `Physical Chemistry`
- Filtro: `Forte`
- Durata spezzoni: `60 secondi`
- Parti italiane escluse: attive
- Parti incerte: attive
- Ripresa automatica: attiva
