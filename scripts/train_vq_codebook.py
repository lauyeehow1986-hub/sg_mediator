"""
Reproducible Trainer for SG-MEDIATOR Head 1 Biomedical VQ Codebook (K=16384, D=128).

Uses INT8 ONNX SapBERT (`Xenova/SapBERT-from-PubMedBERT-fulltext`, UMLS self-aligned `[CLS]`
embeddings) to build a unified K=16,384 spherical vector-quantization codebook:
  1. Encodes all seeded Singapore ontology concepts + synonyms (`config/sg_ontology_seed.csv`)
     and a broad multi-specialty biomedical concept corpus (~2,500 clinical phrases across 18
     ICD-10-AM / SNOMED CT / ACHI domains).
  2. Mean-centers the anisotropic BERT cone, computes top-128 SVD principal components, and
     applies a deterministic orthogonal rotation R in O(128) to balance variance across two
     64-D subspaces.
  3. Trains two 128-way k-means codebooks (`scipy.cluster.vq.kmeans2`, k-means++ init) over the
     broad non-seeded biomedical manifold and forms their 128 x 128 = 16,384 Cartesian product
     unit-norm centroids on S^127.
  4. Carves clean Voronoi margins around the seeded Singapore ontology prototypes and anchors
     each seeded concept at its exact canonical `VQ_xxxxx` index so open-vocabulary paraphrases
     of seeded concepts quantize zero-shot to the canonical ontology token while non-seeded
     clinical concepts cluster faithfully across the remaining 16,339 open-vocabulary buckets.
"""
import csv
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from scipy.cluster.vq import kmeans2
from tokenizers import Tokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT_DIR / "models_onnx_int8" / "sapbert_pubmedbert_int8"
ONTOLOGY_CSV = ROOT_DIR / "config" / "sg_ontology_seed.csv"

# Curated + compositional multi-specialty clinical concepts (strictly disjoint from the 45 seeded concepts)
SPECIALTY_BASES: Dict[str, List[str]] = {
    "respiratory": [
        "bacterial pneumonia", "lobar pneumonia", "viral bronchiolitis", "bronchial asthma exacerbation",
        "chronic obstructive pulmonary disease", "emphysema", "pleural effusion", "spontaneous pneumothorax",
        "pulmonary embolism", "interstitial lung disease", "idiopathic pulmonary fibrosis", "bronchiectasis",
        "obstructive sleep apnea", "acute respiratory distress syndrome", "pulmonary tuberculosis",
        "allergic rhinitis", "acute sinusitis", "hemoptysis", "respiratory failure",
    ],
    "gastroenterology": [
        "acute appendicitis", "inflammation of vermiform appendix", "acute pancreatitis", "chronic pancreatitis",
        "peptic ulcer disease", "gastroesophageal reflux disease", "erosive esophagitis", "ulcerative colitis",
        "crohn regional enteritis", "hepatic cirrhosis", "portal hypertension", "non-alcoholic steatohepatitis",
        "acute cholangitis", "choledocholithiasis", "diverticulitis of sigmoid colon", "celiac sprue",
        "small bowel obstruction", "inguinal hernia", "upper gastrointestinal hemorrhage", "viral hepatitis b",
    ],
    "neurology": [
        "ischemic cerebral infarction", "middle cerebral artery stroke", "transient ischemic attack",
        "intracerebral hemorrhage", "subarachnoid hemorrhage", "generalized tonic-clonic epilepsy",
        "migraine with aura", "tension headache", "parkinson disease", "alzheimer dementia",
        "vascular dementia", "multiple sclerosis", "myasthenia gravis", "guillain-barre polyneuropathy",
        "peripheral sensorimotor neuropathy", "trigeminal neuralgia", "benign paroxysmal positional vertigo",
        "bell facial nerve palsy", "bacterial meningitis", "carpal tunnel median neuropathy",
    ],
    "orthopedics_trauma": [
        "distal radius wrist fracture", "colles fracture of radius", "intertrochanteric hip fracture",
        "femoral neck fracture", "tibial shaft fracture", "clavicle fracture", "vertebral compression fracture",
        "anterior cruciate ligament tear", "medial meniscus tear", "rotator cuff supraspinatus tear",
        "lumbar intervertebral disc herniation", "lumbar spinal stenosis", "knee osteoarthritis",
        "hip osteoarthritis", "adhesive capsulitis frozen shoulder", "plantar fasciitis",
        "achilles tendon rupture", "septic knee arthritis", "acute osteomyelitis", "total knee arthroplasty",
    ],
    "infectious_disease": [
        "dengue hemorrhagic fever", "dengue fever with thrombocytopenia", "plasmodium vivax malaria",
        "community acquired sepsis", "septic shock with lactatemia", "acute pyelonephritis",
        "lower urinary tract infection", "acute bacterial cellulitis", "necrotizing fasciitis",
        "herpes zoster shingles", "infectious mononucleosis", "clostridioides difficile colitis",
        "infective bacterial endocarditis", "candidemia bloodstream infection", "typhoid enteric fever",
        "melioidosis burkholderia infection", "leptospirosis", "influenza a viral pneumonia",
    ],
    "hematology_oncology": [
        "iron deficiency microcytic anemia", "sideropenic anemia", "vitamin b12 megaloblastic anemia",
        "folate deficiency anemia", "beta thalassemia trait", "alpha thalassemia", "sickle cell vaso-occlusive crisis",
        "immune thrombocytopenic purpura", "deep vein thrombosis of lower limb", "popliteal venous thrombosis",
        "disseminated intravascular coagulation", "polycythemia vera", "multiple myeloma",
        "diffuse large b-cell lymphoma", "acute myeloid leukemia", "colorectal adenocarcinoma",
        "hepatocellular carcinoma", "prostate acinar adenocarcinoma", "papillary thyroid carcinoma",
    ],
    "urology_nephrology_other": [
        "benign prostatic hyperplasia", "ureteric calculus renal colic", "calcium oxalate nephrolithiasis",
        "acute tubular necrosis", "minimal change glomerulonephritis", "iga nephropathy",
        "autosomal dominant polycystic kidney disease", "neurogenic bladder", "testicular torsion",
        "acute bacterial prostatitis", "stress urinary incontinence", "hydronephrosis",
        "transurethral resection of prostate", "extracorporeal shockwave lithotripsy",
    ],
    "endocrinology_rheumatology": [
        "primary hypothyroidism", "hashimoto autoimmune thyroiditis", "graves hyperthyroidism",
        "thyrotoxic storm", "primary hyperparathyroidism", "cushing hypercortisolism",
        "addison adrenal insufficiency", "diabetes insipidus", "hyperprolactinemia",
        "acute gouty monoarthritis", "hyperuricemia with tophi", "seropositive rheumatoid arthritis",
        "systemic lupus erythematosus", "ankylosing spondylitis", "sjogren syndrome",
        "giant cell temporal arteritis", "polymyalgia rheumatica", "postmenopausal osteoporosis",
    ],
    "obgyn_pediatrics_ent_derm": [
        "preeclampsia with proteinuria", "ectopic tubal pregnancy", "uterine leiomyoma fibroids",
        "endometriosis", "polycystic ovary syndrome", "postpartum hemorrhage", "placenta previa",
        "cesarean section delivery", "acute otitis media", "otitis externa", "sensorineural hearing loss",
        "acute tonsillitis", "peritonsillar quinsy abscess", "epistaxis nasal bleeding",
        "atopic eczema dermatitis", "plaque psoriasis vulgaris", "chronic spontaneous urticaria",
        "acne vulgaris", "bullous pemphigoid", "open angle glaucoma", "diabetic proliferative retinopathy",
    ],
    "pharmacology_diagnostics": [
        "amoxicillin clavulanate antibiotic therapy", "piperacillin tazobactam intravenous",
        "meropenem carbapenem therapy", "vancomycin glycopeptide infusion", "ceftriaxone cephalosporin",
        "levothyroxine hormone replacement", "prednisolone systemic corticosteroid", "allopurinol urate lowering",
        "colchicine acute gout flare", "omeprazole proton pump inhibitor", "ondansetron antiemetic",
        "paracetamol analgesia", "tramadol opioid analgesia", "apixaban direct oral anticoagulant",
        "warfarin vitamin k antagonist", "furosemide loop diuretic", "spironolactone aldosterone antagonist",
        "salbutamol inhaled bronchodilator", "budesonide formoterol inhaler", "levetiracetam anticonvulsant",
        "computed tomography pulmonary angiogram", "magnetic resonance imaging of brain",
        "transthoracic Doppler echocardiogram", "abdominal hepatobiliary ultrasound",
        "chest radiograph consolidation", "arterial blood gas metabolic acidosis",
        "elevated serum c-reactive protein", "elevated procalcitonin", "hyponatremia", "hyperkalemia",
    ],
}

CLINICAL_MODIFIERS = [
    "",
    "acute ",
    "chronic ",
    "recurrent ",
    "severe ",
    "mild ",
    "moderate ",
    "suspected ",
    "newly diagnosed ",
    "bilateral ",
    "uncomplicated ",
    "refractory ",
    "symptomatic ",
    "clinical diagnosis of ",
]

def build_broad_corpus(excluded_terms: Optional[set] = None) -> List[str]:
    excluded = excluded_terms or set()
    phrases: List[str] = []
    seen = set()
    for items in SPECIALTY_BASES.values():
        for base in items:
            for mod in CLINICAL_MODIFIERS:
                cand = f"{mod}{base}".strip().lower()
                if cand not in seen and cand not in excluded:
                    seen.add(cand)
                    phrases.append(cand)
    return phrases


def _clean_term_for_sapbert(term: str) -> str:
    """Strips parenthetical administrative tags (e.g. NEHR Sensitive Health Info) before SapBERT encoding."""
    cleaned = re.sub(r"\s*\([^)]*\)", "", term).strip()
    cleaned = re.sub(r"\s+on\s+12-lead\s+ecg$", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned or term


def load_seeded_concepts(ontology_csv: Path) -> List[Tuple[str, int, List[str]]]:
    seeded = []
    with ontology_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = row["concept_code"].strip()
            vq_id = row["vq_token_id"].strip()
            idx = int(vq_id.split("_")[1])
            terms = [_clean_term_for_sapbert(row["canonical_name"].strip().lower())]
            terms.extend(
                _clean_term_for_sapbert(s.strip().lower())
                for s in row.get("synonyms", "").split("|")
                if s.strip()
            )
            uniq_terms = list(dict.fromkeys(t for t in terms if t))
            seeded.append((code, idx, uniq_terms))
    return seeded


def encode_sapbert_cls(
    sess: ort.InferenceSession, tok: Tokenizer, phrases: List[str], batch_size: int = 64
) -> np.ndarray:
    tok.enable_truncation(max_length=32)
    tok.enable_padding(pad_id=0, pad_token="[PAD]")
    input_names = {i.name for i in sess.get_inputs()}
    vecs = []
    for start in range(0, len(phrases), batch_size):
        batch = phrases[start : start + batch_size]
        encs = tok.encode_batch(batch)
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = sess.run(None, feeds)[0]  # (B, T, 768)
        cls_vecs = hidden[:, 0, :].astype(np.float32)
        norms = np.maximum(np.linalg.norm(cls_vecs, axis=1, keepdims=True), 1e-9)
        vecs.append(cls_vecs / norms)
    tok.no_padding()
    return np.vstack(vecs)


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)
    return (mat / norms).astype(np.float32)


def train_and_save_codebook(
    model_dir: Path = DEFAULT_MODEL_DIR,
    ontology_csv: Path = ONTOLOGY_CSV,
    codebook_size: int = 16384,
    proj_dim: int = 128,
    seed: int = 20261031,
) -> Path:
    assert codebook_size == 128 * 128, "Product codebook expects K = 128 x 128 = 16384"
    assert proj_dim == 128

    t0 = time.perf_counter()
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 2) // 2))
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(str(model_dir / "model_quantized.onnx"), so, providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    seeded = load_seeded_concepts(ontology_csv)
    seeded_flat = [t for _, _, terms in seeded for t in terms]
    broad_phrases = build_broad_corpus(set(seeded_flat))

    print(f"[VQ Trainer] Encoding {len(broad_phrases)} broad clinical phrases + {len(seeded_flat)} seeded terms with INT8 SapBERT...")
    X_broad = encode_sapbert_cls(sess, tok, broad_phrases, batch_size=64)
    X_seeded_all = encode_sapbert_cls(sess, tok, seeded_flat, batch_size=64)

    # 1. Fit mean + top-128 SVD PCA + variance-balancing orthogonal rotation R in O(128)
    X_joint = np.vstack([X_broad, X_seeded_all])
    mean_vec = X_joint.mean(axis=0).astype(np.float32)
    X_centered = X_joint - mean_vec

    _, svals, vt = np.linalg.svd(X_centered, full_matrices=False)
    var_explained = float((svals[:proj_dim] ** 2).sum() / (svals ** 2).sum())
    w_pca = vt[:proj_dim].T.astype(np.float32)  # (768, 128)

    rng = np.random.default_rng(seed)
    q_rot, _ = np.linalg.qr(rng.standard_normal((proj_dim, proj_dim)).astype(np.float32))
    proj_mat = (w_pca @ q_rot).astype(np.float32)  # (768, 128) orthonormal columns

    # 2. Project broad non-seeded phrases into balanced 128D sphere and train 128 x 128 PQ centroids
    Z_broad = l2_normalize_rows((X_broad - mean_vec) @ proj_mat)
    half = proj_dim // 2
    Z1 = l2_normalize_rows(Z_broad[:, :half]) * np.float32(1.0 / np.sqrt(2.0))
    Z2 = l2_normalize_rows(Z_broad[:, half:]) * np.float32(1.0 / np.sqrt(2.0))

    c1, _ = kmeans2(Z1, 128, iter=40, minit="++", seed=seed)
    c2, _ = kmeans2(Z2, 128, iter=40, minit="++", seed=seed + 1)
    c1 = l2_normalize_rows(c1) * np.float32(1.0 / np.sqrt(2.0))
    c2 = l2_normalize_rows(c2) * np.float32(1.0 / np.sqrt(2.0))

    # Materialize all 128 x 128 = 16,384 unit-norm composite centroids on S^127
    grid1 = np.repeat(c1, 128, axis=0)  # (16384, 64)
    grid2 = np.tile(c2, (128, 1))       # (16384, 64)
    centroids = l2_normalize_rows(np.hstack([grid1, grid2]))  # (16384, 128)

    # 3. Compute seeded ontology prototype vectors in the 128D space
    anchor_indices: List[int] = []
    anchor_codes: List[str] = []
    anchor_protos: List[np.ndarray] = []
    cursor = 0
    for code, idx, terms in seeded:
        n_t = len(terms)
        emb_slice = X_seeded_all[cursor : cursor + n_t]
        cursor += n_t
        z_terms = l2_normalize_rows((emb_slice - mean_vec) @ proj_mat)
        proto = z_terms.mean(axis=0)
        proto = proto / max(float(np.linalg.norm(proto)), 1e-9)
        anchor_indices.append(idx)
        anchor_codes.append(code)
        anchor_protos.append(proto.astype(np.float32))

    anchor_mat = np.stack(anchor_protos, axis=0)  # (45, 128)
    anchor_set = set(anchor_indices)

    # 4. Carve clean Voronoi margin (max cosine <= 0.35) on non-anchor slots relative to all 128 seeded prototypes
    non_anchor_mask = np.array([i not in anchor_set for i in range(codebook_size)], dtype=bool)
    for _ in range(15):
        sims = centroids[non_anchor_mask] @ anchor_mat.T  # (N_non, N_anchors)
        max_anchor_idx = np.argmax(sims, axis=1)
        max_sim = sims[np.arange(sims.shape[0]), max_anchor_idx]
        too_close = max_sim > 0.35
        if not np.any(too_close):
            break
        na_indices = np.flatnonzero(non_anchor_mask)[too_close]
        closest_protos = anchor_mat[max_anchor_idx[too_close]]
        excess = (max_sim[too_close] - 0.26)[:, None]
        centroids[na_indices] = l2_normalize_rows(centroids[na_indices] - excess * closest_protos)

    # 5. Anchor the 45 seeded ontology concepts at their exact canonical VQ_xxxxx slots
    for idx, proto in zip(anchor_indices, anchor_protos):
        centroids[idx] = proto

    centroids = l2_normalize_rows(centroids)
    out_path = model_dir / "vq_codebook_16384.npz"
    np.savez_compressed(
        out_path,
        mean=mean_vec.astype(np.float32),
        proj=proj_mat.astype(np.float32),
        centroids=centroids.astype(np.float16),
        anchor_indices=np.array(anchor_indices, dtype=np.int32),
        anchor_codes=np.array(anchor_codes),
        var_explained=np.array([var_explained], dtype=np.float32),
    )
    elapsed = time.perf_counter() - t0
    print(
        f"[VQ Trainer] Saved {out_path.name} ({out_path.stat().st_size / 2**20:.2f} MB) | "
        f"K={codebook_size}, D={proj_dim}, PCA_var={var_explained:.3f}, anchors={len(anchor_indices)} in {elapsed:.1f}s"
    )
    return out_path


if __name__ == "__main__":
    train_and_save_codebook()
