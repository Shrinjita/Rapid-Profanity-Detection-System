# **Multilingual Profanity Detection System (MPDS)**

## **Overview**

**MPDS** is a multilingual, low-latency profanity detection system designed as a **plugin-style module** for gaming and interactive applications.
It integrates via **REST API** or **WebSocket**, enabling asynchronous, real-time moderation without embedding full detection logic inside the game client.
The repository includes three sequential development stages — **v0**, **v1**, and **v2** — each representing a milestone in the system’s evolution.

---

## **V0 — Prototype**

**Purpose:**
Initial proof-of-concept to test feasibility of profanity detection in a lightweight environment.

**Location:** `v0/`

**Contents:**

* `app.py` – Streamlit prototype for text-based profanity filtering.
* `detector.py` – Basic rule-based detection algorithm.
* `dataset_small.csv` – Minimal demo dataset.
* `generate_dataset.py` – Small dataset builder utility.
* `demo_script.txt` – Sample text input for testing.

**Summary:**

* Served as the experimental foundation for model behavior.
* Built for internal validation and classroom demonstration.
* Retained for quickstart reference only.

---

## **V1 — MVP (Current Stable Version)**

**Purpose:**
Functional Minimum Viable Product integrating a working frontend and backend with real-time moderation inside a game loop.

**Location:** `v1/`

**Key Components:**

* `main.py` – Flask-SocketIO backend for handling in-game text events.
* `templates/game.html` – Space Defender demo (reference game).
* `static/script.js`, `static/style.css` – Client logic and visuals.

**Architecture:**

* Uses **Flask + SocketIO** for bidirectional communication.
* Demonstrates MPDS integration as a **WebSocket plugin** in an active game.
* Acts as the **reference implementation**, not the final production model.

**Datasets:**

* Uses **test strings only**; no standalone dataset files.
* All curated datasets reside in **`v2/`** for upcoming model versions.

**Performance Baseline:**

* Latency: ~200–300 ms per input.
* Accuracy: ~80–85 % on text.
  *(These serve as baseline benchmarks for future comparison.)*

**Status:**
Stable and fully functional MVP used for demonstrations and evaluation.

---

## **V2 — Under Development**

**Purpose:**
Algorithmic upgrade emphasizing **speed**, **obfuscation tolerance**, and **multilingual support**.

**Location:** `v2/`

**Core Enhancements:**

* **Bounded-Edit Aho–Corasick Algorithm**

  * Fused canonicalization and matching for single-pass processing.
  * Handles homoglyphs, transliterations, and leetspeak variants.
* **Full Canonicalizer**

  * Unicode NFKC normalization, punctuation stripping, case folding.
* **Offline Speech-to-Text (STT)** using **Vosk**

  * English: `vosk-model-small-en-us-0.15`
  * Hindi: `vosk-model-small-hi-0.22`
  * Models are **user-downloaded**, not bundled (~220 MB total).
* **Datasets:**

  * `dataset_primary.txt` – proprietary, hand-curated.
  * `dataset_secondary.txt` – aggregated from public GitHub lists.
* **Logging:**

  * `logs/profanity_events.log` for event tracking and escalation.
* **Warning Logic:**

  * Three-stage escalation → pop-up alerts → session shutdown trigger.

**Execution Modes:**

* `text_only` – runs profanity detection only.
* `vosk` – activates offline speech recognition.
* `streamlit` – placeholder, currently inactive.

**Deployment Intent:**

* To operate as a **local plugin** via REST or WebSocket interface.
* Designed for **self-hosted** or private server integration.

**Status:**
Experimental; in production phase with focus on optimization and multilingual scaling.

---

## **Architecture Diagram**

<img width="1029" height="1109" alt="TLRDZXj54BvRyZkKWaGs2fr0eXpm0SNnjdUj-4-UsGMK8DGxqvvfkQVRwkxPnK58NE40XpqH2P5396wSEV4m-mBm25HtpnXxlOmborl_NzLNDVlSEcPSNSbkvpwaF2kLo9W4ooj1dXw3tLZ7AyW5AmohuAXaSabARSGRhHpQXRylkN71AwilE5YtaVn1jzFjs9LGQ-RzBbYsA" src="https://github.com/user-attachments/assets/e6df2d99-b228-4a16-aedd-ab063bccd8a3" />


---

## **Future Direction**

* Extend coverage across **major Indian languages** (English, Hindi, Bengali initially).
* Improve false-positive handling and efficiency in streaming contexts.
* Package MPDS as a deployable **cross-engine plugin library**.
