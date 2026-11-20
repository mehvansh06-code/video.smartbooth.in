# SmartBooth AI Video Campaign Engine (`video.smartbooth.in`)
**High-throughput, hyper-personalized political campaign video rendering pipeline for the Google Hackathon.**

🌐 **Live URL**: [https://video.smartbooth.in](https://video.smartbooth.in)  
☁️ **Infrastructure**: Google Cloud Compute Engine (`smartbooth-demo` / `8.231.107.200`) proxied with Cloudflare Full SSL.

---

## 🚀 Key Features
- **Surgical Splice Architecture**: Merges personalized candidate voice greeting synthesized via ElevenLabs multilingual AI (`eleven_multilingual_v2`) with authentic speech body at calibrated acoustic cuts.
- **Dynamic Devanagari Graphics**: Pillow-rendered voter honorific recognition badge with election symbol and ballot banner.
- **WhatsApp Optimized**: Enforces strict 720x1280 @ 30fps H.264 (< 3.3MB) with `+faststart` moov atom for immediate mobile streaming.
- **Interactive Multi-Voter Showcase**: Ingests campaign video and renders personalized video variants for 4 to 5 voter names with 1-click WhatsApp sharing and instant MP4 downloads.
- **Pre-Flight Compliance Auditor**: Built-in verification testing resolution, codecs, AAC 48kHz standard, and mobile bitrate.

---

## 🛠️ Tech Stack
- **Backend**: Python 3, Flask, Gunicorn
- **Audio & Video Engine**: FFmpeg, ElevenLabs Multilingual v2 API
- **Typography & Canvas**: Pillow, Google Noto Sans Devanagari
- **Web Infrastructure**: Nginx, Systemd, Google Cloud Platform, Cloudflare CDN
