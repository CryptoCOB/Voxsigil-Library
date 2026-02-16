"""
Generate diverse, unique sigil patterns (3-8 unique unicode chars each).
Updates all 35,823 sigils with proper glyph combinations.
"""

import sys
import random
from pathlib import Path

import yaml

OUTPUT_BASE = Path("c:/nebula-social-crypto-core/voxsigil_library/library_sigil_enhanced")

# Unicode character pools for diverse sigil creation
MATHEMATICAL = "∀∃∄∅∇∈∉∊∋∌∍∎∏∐∑−∗√∛∜∝∞∟∠∡∢∣∤∥∦∧∨∩∪≀≁≂≃≄≅≆≇≈≉≊≋≌≍≎≏≐≑≒≓≔≕≖≗≘≙≚≛≜≝≞≟≠≡≢≣≤≥≦≧≨≩≪≫≬≭≮≯≰≱≲≳≴≵≶≷≸≹≺≻≼≽"
ARROWS = "←↑→↓↔↕↖↗↘↙↚↛↜↝↞↟↠↡↢↣↤↥↦↧↨↩↪↫↬↭↮↯↰↱↲↳↴↵↶↷↸↹↺↻↼↽↾↿⇀⇁⇂⇃⇄⇅⇆⇇⇈⇉⇊⇋⇌⇍⇎⇏⇐⇑⇒⇓⇔⇕⇖⇗⇘⇙⇚⇛⇜⇝⇞"
GEOMETRIC = "◀▶◀▶■□▪▫●○◐◑◒◓◔◕◖◗◘◙◚◛◜◝◞◟◠◡◢◣◤◥◦◧◨◩◪◫◬◭◮◯◰◱◲◳◴◵◶◷◸◹◺◻◼◽◾◿"
SYMBOLS = "✁✂✃✄✅✆✇✈✉✊✋✌✍✎✏✐✑✒✓✔✕✖✗✘✙✚✛✜✝✞✟✠✡✢✣✤✥✦✧✨✩✪✫✬✭✮✯✰✱✲✳✴✵✶✷✸✹✺✻✼✽✾✿❀"
BRACKETS = "⟨⟩⟪⟫⟬⟭⟮⟯⟰⟱⟲⟳⟴⟵⟶⟷⟸⟹⟺⟻⟼⟽⟾⟿"
FLOURISH = "❍❎❏❐❑❒❓❔❕❖❗❘❙❚❛❜❝❞❡❢❣❤❥❦❧❨❩❪❫❬❭❮❯❰❱❲❳❴❵❶❷❸❹❺❻❼❽❾❿"
STARS_BOXES = "★☆✦✧✩✪✫✬✭✮✯✱◈◆◇◈◉◊○●◐◑◒◓"
EGYPTIAN = "𓀀𓀁𓀂𓀃𓀄𓀅𓀆𓀇𓀈𓀉𓀊𓀋𓀌𓀍𓀎𓀏𓀐𓀑𓀒𓀓𓀔𓀕𓀖𓀗𓀘𓀙𓀚𓀛𓀜𓀝𓀞𓀟"

POOLS = [MATHEMATICAL, ARROWS, GEOMETRIC, SYMBOLS, BRACKETS, FLOURISH, STARS_BOXES, EGYPTIAN]


def generate_unique_sigil() -> str:
    """Generate a 3-8 character sigil with unique unicode glyphs."""
    # Random count between 3 and 8
    count = random.randint(3, 8)
    
    # Select random character pools (can repeat pools but not chars)
    sigil_chars = []
    used_chars = set()
    
    # Try to get unique characters
    attempts = 0
    max_attempts = 50
    
    while len(sigil_chars) < count and attempts < max_attempts:
        pool = random.choice(POOLS)
        char = random.choice(pool)
        if char not in used_chars:
            sigil_chars.append(char)
            used_chars.add(char)
        attempts += 1
    
    return "".join(sigil_chars)


def update_sigil_in_file(file_path: Path) -> bool:
    """Update sigil field in a file."""
    try:
        # Read
        text = None
        for encoding in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
            try:
                text = file_path.read_text(encoding=encoding, errors="ignore")
                break
            except Exception:
                continue
        
        if not text:
            return False
        
        # Parse
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            return False
        
        # Update sigil
        if "meta" not in data or not isinstance(data["meta"], dict):
            return False
        
        data["meta"]["sigil"] = generate_unique_sigil()
        
        # Write back
        yaml_str = yaml.dump(
            data,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        file_path.write_text(yaml_str, encoding="utf-8")
        return True
    
    except Exception:
        return False


def main():
    """Update all sigils in enhanced corpus."""
    sigil_files = list((OUTPUT_BASE / "sigils").glob("*.voxsigil"))
    total = len(sigil_files)
    
    if total == 0:
        print("[!] No sigil files found")
        return
    
    successes = 0
    failures = 0
    
    print(f"[*] Generating diverse sigils (3-8 unique chars)...")
    print(f"[*] Processing {total} sigil files...\n")
    sys.stdout.flush()
    
    for idx, fpath in enumerate(sigil_files):
        if update_sigil_in_file(fpath):
            successes += 1
        else:
            failures += 1
        
        if (idx + 1) % 1000 == 0:
            pct = 100 * (idx + 1) / total
            print(f"[*] {idx + 1}/{total} ({pct:.1f}%)")
            sys.stdout.flush()
    
    print("\n" + "=" * 60)
    print(f"SIGILS UPDATED: {successes}/{total}")
    print(f"ERRORS:         {failures}")
    print(f"SUCCESS RATE:   {100*successes/total:.1f}%")
    print("=" * 60)
    print("\n[✓] Sigils now use 3-8 unique unicode characters each!")


if __name__ == "__main__":
    main()
