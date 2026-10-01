# Application handoff

## Resume bullets — choose two, not all

- Built a reproducible manufacturing quality-triage pipeline on 1,567 semiconductor production records using Python and scikit-learn, with chronological evaluation, training-only preprocessing, and exact logistic-model explanations.
- Evaluated a fixed 20% review policy on 314 held-out units, capturing 7 of 17 failures versus 3.4 expected under random review; documented 56 false alarms and inconsistent validation performance.
- Integrated local Ollama into evidence-grounded unit and batch investigation briefs with schema validation, traceable citations, deterministic fallback, and SQLite generation/review logs.

Use the measured performance bullet only if you can explain the zero-capture validation result and why this is a prototype. Do not claim operational savings, calibrated probabilities, physical root-cause diagnosis, OpenAI API integration, or production adoption. Human-review status and hosting should be represented exactly as verified at handoff.

## Two-minute walkthrough

**0:00–0:20 — Business question.** “I wanted to explore how a quality team could prioritize limited review capacity, using public semiconductor manufacturing records. This is not Seagate's data.”

**0:20–0:45 — Data and leakage controls.** Show missingness, anonymous measurement names, chronological partitions, training-only preprocessing, and the frozen run. “I avoided random row splitting and kept outcomes out of the LLM inputs.”

**0:45–1:10 — Review workflow.** Inspect a ranked unit, positive/negative contributions, missing inputs and a reference distribution. “These contributions explain the model, not the physical cause. Extreme values can make the score saturate.”

**1:10–1:35 — Grounded AI.** Show a unit and batch brief, evidence references, model/prompt identity, and review status. “Python provides the numerical facts; Ollama drafts cautious connective prose. Unsupported output is rejected or replaced with a labeled template.”

**1:35–2:00 — Results and limitations.** “The default queue captured seven of seventeen test failures while flagging fifty-six passes. Validation captured no failures, so I would not deploy this model operationally. The important result is a reproducible, inspectable workflow and a clear account of where it fails.”

## Thai follow-up draft to Rasita — not sent

ขอบคุณสำหรับคำแนะนำมากครับพี่รสิตา ผมลองนำคำแนะนำของพี่มาทำโปรเจกต์จากข้อมูลการผลิต semiconductor ที่เปิดเผยสาธารณะของ SECOM ครับ โดยทำตั้งแต่ตรวจสอบข้อมูลและ missing values ไปจนถึงโมเดลจัดลำดับชิ้นงานที่ควรตรวจสอบ และใช้ local LLM ช่วยร่างสรุปหลักฐานสำหรับการตรวจสอบครับ

ผมใส่ทั้งผลการประเมินและข้อจำกัดไว้ด้วย เพราะผลระหว่างช่วง validation กับ test ยังไม่สม่ำเสมอ และข้อมูลไม่ได้ระบุความหมายของแต่ละ measurement เลยยังไม่สามารถสรุปสาเหตุทางกายภาพได้ครับ

นี่คือลิงก์โปรเจกต์ครับ: https://github.com/beam-9/manufacturing-quality-triage-SECOM

ถ้าพี่มีเวลาสะดวก ผมรบกวนขอคำแนะนำได้ไหมครับว่าแนวทางนี้ช่วยแสดงความสนใจด้าน manufacturing และเหมาะกับการสมัคร AI Adoption Intern มากขึ้นไหมครับ ขอบคุณพี่อีกครั้งครับ ไม่ต้องรีบตอบเลยครับ

Add the public demo URL only after verification. This draft asks for feedback, not confidential information or an assumed referral. A referral-process question can follow naturally if she responds positively.

