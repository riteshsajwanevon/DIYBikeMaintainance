# Feature: Check Engine Oil Level and Refill

This document outlines the architecture for the "Check Engine Oil Level" feature, demonstrating the interaction between the Computer Vision (CV) model, LangGraph (State Tracking), and LangChain (LLM Recommendations).

## 1. LangGraph State Machine (Workflow)

The repair process is broken down into a state graph with the following nodes (steps):

1. **`Locate_Dipstick`**: Find and remove the oil dipstick.
2. **`Wipe_Dipstick`**: Clean the dipstick with a rag.
3. **`Reinsert_And_Check`**: Put the dipstick back in and pull it out to read the level.
4. **`Evaluate_Level`**: (Decision Node) 
   - If Low -> Go to `Locate_Oil_Cap`.
   - If Good -> Go to `Complete`.
5. **`Locate_Oil_Cap`**: Find and remove the engine oil filler cap.
6. **`Pour_Oil`**: Insert funnel and pour new oil.
7. **`Recheck_Level`**: Go back to `Wipe_Dipstick` to ensure the correct level is reached.

## 2. Object Detection Requirements

The CV model needs to be trained to detect the following objects (classes):
- `dipstick_handle`
- `dipstick_blade`
- `rag` / `paper_towel`
- `oil_filler_cap`
- `funnel`
- `engine_oil_bottle`
- `washer_fluid_bottle` (for error checking)

## 3. Converting Raw CV Data to LLM Input

Object detection models (like YOLO) do not output clean JSON strings. They output raw arrays of numbers representing bounding boxes, class indices, and confidence scores. 
To convert this raw data into the JSON payloads seen below, you need a middleware script (usually in Python).

### The Pipeline

1.  **Raw CV Output (The Numbers):**
    For each frame, the CV model outputs an array like: `[x_min, y_min, x_max, y_max, confidence, class_id]`
    *Example:* `[120, 45, 180, 210, 0.95, 2]` (where class `2` is `dipstick_blade`)

2.  **Mapping and Filtering (The Logic):**
    A Python script intercepts this data:
    - It maps `class_id 2` to the string `"dipstick_blade"`.
    - It filters out low-confidence detections (e.g., ignoring anything below `0.50`).

3.  **Spatial Reasoning (The Math):**
    The script calculates the overlap (Intersection over Union, or IoU) between bounding boxes.
    - If the `rag` bounding box overlaps significantly with the `dipstick_blade` bounding box, the script infers they are interacting.
    - It translates this math into a plain text string: `"The rag is overlapping with the dipstick_blade."`

4.  **State Injection:**
    The script polls **LangGraph** to ask, "What step are we currently on?". LangGraph replies: `"Wipe_Dipstick"`.

5.  **Payload Assembly (The Final JSON):**
    The script takes the mapped labels, the spatial math, and the LangGraph state, and assembles them into a structured JSON dictionary. This JSON is then sent to the LLM (LangChain) for decision making.

## 4. Example Scenarios (JSON Interface)

### Scenario A: User is wiping the dipstick (Correct Action)

**Context:** The user just pulled out the dipstick and needs to wipe it clean before getting an accurate reading.
**Current Graph State:** `Wipe_Dipstick`

**1. Input from CV to LLM:**
```json
{
  "current_task": "Check Engine Oil",
  "expected_step": "Wipe the dipstick clean",
  "detected_objects": [
    {"label": "dipstick_blade", "confidence": 0.95, "status": "in_hand"},
    {"label": "rag", "confidence": 0.90, "status": "in_hand"}
  ],
  "spatial_context": "The rag is overlapping with the dipstick_blade."
}
```

**2. Output from LLM to Application:**
```json
{
  "user_message": "Good job. Wipe the oil off completely. Once it's clean, insert it fully back into the tube and pull it out again to check the level.",
  "step_status": "completed",
  "next_state": "Reinsert_And_Check",
  "warnings": null
}
```

> [!TIP]
> Notice how the LLM translates the raw object data ("rag overlapping dipstick") into a state change ("completed") and a human-friendly instruction ("Good job. Wipe the oil off...").

### Scenario B: Oil is low, user grabs the wrong fluid

**Context:** The dipstick showed low oil. The graph moved to the refill stage. The user is about to pour fluid, but the camera detects windshield washer fluid instead of engine oil.
**Current Graph State:** `Pour_Oil`

**1. Input from CV to LLM:**
```json
{
  "current_task": "Check Engine Oil",
  "expected_step": "Pour engine oil into the filler hole",
  "detected_objects": [
    {"label": "oil_filler_cap", "status": "removed"},
    {"label": "funnel", "status": "inserted_in_engine"},
    {"label": "washer_fluid_bottle", "confidence": 0.98, "status": "in_hand"}
  ],
  "spatial_context": "The washer_fluid_bottle is positioned above the funnel."
}
```

**2. Output from LLM to Application:**
```json
{
  "user_message": "STOP! You are holding Windshield Washer Fluid. Do not pour that into the engine, it will cause severe damage. Please grab the Engine Oil bottle.",
  "step_status": "error",
  "next_state": "Pour_Oil",
  "warnings": "CRITICAL: Wrong fluid detected (washer_fluid_bottle). Alerting user to stop immediately."
}
```

> [!CAUTION]
> This demonstrates the safety power of combining CV with an LLM. The LLM understands the semantic danger of mixing these fluids and throws a critical warning while preventing the state machine from advancing.

### Scenario C: Refilling the Oil (Correct Action)

**Context:** The user correctly swapped to the engine oil bottle.
**Current Graph State:** `Pour_Oil`

**1. Input from CV to LLM:**
```json
{
  "current_task": "Check Engine Oil",
  "expected_step": "Pour engine oil into the filler hole",
  "detected_objects": [
    {"label": "funnel", "status": "inserted"},
    {"label": "engine_oil_bottle", "confidence": 0.94, "status": "in_hand", "orientation": "tilted_down"}
  ],
  "spatial_context": "The engine_oil_bottle is pouring into the funnel."
}
```

**2. Output from LLM to Application:**
```json
{
  "user_message": "Perfect. Pour a small amount, about a quarter of a quart, then stop. We need to recheck the dipstick so we don't overfill it.",
  "step_status": "in_progress",
  "next_state": "Recheck_Level",
  "warnings": null
}
```
