# AgriCare 🌱  
AI-powered crop disease detection and advisory tool for Nigerian farmers.  

## Overview  
AgriCare helps farmers detect crop diseases using AI and provides **personalized advice** based on:  
- **Crop health diagnosis** from leaf images (deep learning, EfficientNet).  
- **Farm size** (Smallholder vs. Large farm) → short-term vs. long-term recommendations.  
- **Language options**: English, Hausa, and Yoruba (text + audio using gTTS).  
- **Human-in-the-loop review** for low-confidence predictions.  

Unlike global apps (Plantix, Agrio), AgriCare is built for **local Nigerian context** with **offline readiness (future upgrade)** and **multilingual farmer support**.  

---

## Features  
- 📷 **Image-based disease detection** (cassava dataset for prototype).  
- 🔊 **Multilingual audio advice** (English, Hausa, Yoruba).  
- 🧑‍🌾 **Tailored advice by farm size** (small vs. large farms).  
- ⚠️ **Human review queue** for uncertain cases.  
- 🌍 **Scalable to other Nigerian staple crops** (maize, yam, rice).  

---

## Installation  

1. Clone the repository:  
```bash
git clone https://github.com/yourusername/agricare.git
cd agricare