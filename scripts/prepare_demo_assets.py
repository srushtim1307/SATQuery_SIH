import os
from pathlib import Path
from PIL import Image

def extract_demo_assets():
    base_dir = Path(__file__).resolve().parent.parent
    demo_out = base_dir / "frontend" / "public" / "demo"
    demo_out.mkdir(parents=True, exist_ok=True)
    
    # 1. Extract from SatQuery AI - Antigravity Workspace.png
    ws_path = base_dir / "SatQuery AI - Antigravity Workspace.png"
    if ws_path.exists():
        ws = Image.open(ws_path).convert("RGB")
        # Estuary delta scene
        estuary = ws.crop((360, 890, 1188, 1360))
        estuary.save(demo_out / "sentinel2_estuary_delta.png", "PNG")
        
        # Before acquisition (May 2023)
        before = ws.crop((360, 1715, 764, 2025))
        before.save(demo_out / "sentinel2_may2023.png", "PNG")
        
        # After acquisition (Oct 2024)
        after = ws.crop((783, 1715, 1188, 2025))
        after.save(demo_out / "sentinel2_oct2024.png", "PNG")
        print("Extracted Workspace assets: estuary, before, after")

    # 2. Extract from SatQuery AI - Analysis History.png
    hist_path = base_dir / "SatQuery AI - Analysis History.png"
    if hist_path.exists():
        hist = Image.open(hist_path).convert("RGB")
        # Card 1: Urban Expansion (2B badge)
        thumb1 = hist.crop((352, 354, 432, 434))
        thumb1.save(demo_out / "thumb_urban_expansion.png", "PNG")
        
        # Card 2: Water Body (1B badge)
        thumb2 = hist.crop((352, 517, 432, 597))
        thumb2.save(demo_out / "thumb_water_body.png", "PNG")
        
        # Card 3: Optical + SAR (SAR badge)
        thumb3 = hist.crop((352, 680, 432, 760))
        thumb3.save(demo_out / "thumb_optical_sar.png", "PNG")
        
        # Card 4: Agricultural Land Use (NDVI badge)
        thumb4 = hist.crop((352, 866, 432, 946))
        thumb4.save(demo_out / "thumb_agri_ndvi.png", "PNG")
        
        # Card 5: Coastal Flood (DEM badge)
        thumb5 = hist.crop((352, 1052, 432, 1132))
        thumb5.save(demo_out / "thumb_coastal_dem.png", "PNG")
        print("Extracted History thumbnails")

    # 3. Extract SAR scene from SatQuery AI - New Analysis (Antigravity Style).png
    na_path = base_dir / "SatQuery AI - New Analysis (Antigravity Style).png"
    if na_path.exists():
        na = Image.open(na_path).convert("RGB")
        # Staged SAR thumbnail
        sar_thumb = na.crop((365, 735, 415, 785))
        sar_thumb.save(demo_out / "sar_thumb.png", "PNG")
        
        # Also staged estuary thumbnail
        est_thumb = na.crop((365, 655, 415, 705))
        est_thumb.save(demo_out / "est_thumb.png", "PNG")
        print("Extracted New Analysis thumbnails")

if __name__ == "__main__":
    extract_demo_assets()
