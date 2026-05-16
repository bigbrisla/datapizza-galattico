import json

from .schemas import EXTRACTION_JSON_SCHEMA, MONGO_QUERY_JSON_SCHEMA


def metadata_extraction_prompt(
    source_document: str,
    chunk_text: str,
    known_vocabulary: dict[str, list[str]],
) -> str:
    return f"""
Sei un sistema di estrazione dati. Estrai metadati strutturati dal testo fornito.

Regole:
- Non inventare informazioni.
- Se un campo non e presente nel testo, usa null o lista vuota.
- Riusa i valori del vocabolario esistente quando indicano lo stesso concetto.
- Mantieni anche il testo originale utile per verifiche successive.
- Rispondi solo con JSON valido conforme allo schema.

Documento sorgente:
{source_document}

Vocabolario esistente:
{json.dumps(known_vocabulary, ensure_ascii=False, indent=2)}

Schema JSON:
{json.dumps(EXTRACTION_JSON_SCHEMA, ensure_ascii=False, indent=2)}

Testo:
{chunk_text}
""".strip()


def query_transform_prompt(
    question: str,
    available_fields: list[str],
    known_vocabulary: dict[str, list[str]],
    allowed_fields: list[str] | None = None,
    generic_terms_to_avoid: list[str] | None = None,
) -> str:
    allowed_fields = allowed_fields or available_fields
    generic_terms_to_avoid = generic_terms_to_avoid or []
    return f"""
Sei un esperto di MongoDB. Genera una query MongoDB JSON per recuperare i documenti utili.

Regole:
- Non inventare campi.
- Usa solo campi disponibili.
- Usa solo valori presenti nel vocabolario quando possibile.
- Se la query e troppo incerta, usa condizioni parziali.
- Preferisci query selettive sui metadati; non usare vector search.
- Se la domanda contiene un target esplicito dopo verbi come "contiene", "include", "usa", "utilizza",
  quel target va trattato come vincolo principale su ingredients o techniques.
- Non sostituire il target esplicito con parole di contesto generico (es. "galassia", "piatti", "ristorante").
- Mantieni i token speciali dei target letterali (es. "+", apostrofi, trattini) quando presenti.
- Se e presente un target esplicito ingrediente/tecnica:
  - non aggiungere filtri semantici o stilistici su dish_name/restaurant;
  - non usare regex narrative (es. "cosmica", "galattica", "magico", "celestiale") non richieste esplicitamente;
  - preferisci una query minima con soli vincoli espliciti.
- Usa solo i campi ammessi per questo caso.
- Evita di usare come vincoli termini generici/non discriminativi.
- Rispondi solo con JSON valido.

Campi disponibili:
{json.dumps(available_fields, ensure_ascii=False, indent=2)}

Campi ammessi per questo caso:
{json.dumps(allowed_fields, ensure_ascii=False, indent=2)}

Termini generici da evitare come vincoli:
{json.dumps(generic_terms_to_avoid, ensure_ascii=False, indent=2)}

Vocabolario:
{json.dumps(known_vocabulary, ensure_ascii=False, indent=2)}

Schema output:
{json.dumps(MONGO_QUERY_JSON_SCHEMA, ensure_ascii=False, indent=2)}

Domanda:
{question}
""".strip()


def query_retry_prompt(question: str, previous_query: dict, reason: str, known_vocabulary: dict[str, list[str]]) -> str:
    return f"""
La query MongoDB precedente non ha funzionato.

Domanda:
{question}

Query precedente:
{json.dumps(previous_query, ensure_ascii=False, indent=2)}

Motivo:
{reason}

Vocabolario disponibile:
{json.dumps(known_vocabulary, ensure_ascii=False, indent=2)}

Genera una nuova query JSON piu adatta.
Rispondi solo con JSON valido nello stesso formato:
{json.dumps(MONGO_QUERY_JSON_SCHEMA, ensure_ascii=False, indent=2)}
""".strip()


def candidate_rerank_prompt(question: str, candidates: list[dict]) -> str:
    schema = {
        "type": "object",
        "properties": {
            "dish_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Lista dei dish_id corretti per la domanda.",
            }
        },
        "required": ["dish_ids"],
    }
    return f"""
Sei un validatore di matching domanda-piatti.

Regole:
- Valuta solo i candidati forniti.
- Se un vincolo e esplicito (pianeta, ingrediente, tecnica, licenza, grado minimo), rispettalo in modo rigoroso.
- Non inventare piatti non presenti nella lista candidati.
- Se nessun candidato e valido, restituisci dish_ids vuoto.
- Rispondi solo con JSON valido.

Domanda:
{question}

Candidati:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Schema output:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()
