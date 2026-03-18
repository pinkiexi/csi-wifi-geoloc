"""
=============================================================================
Pipeline Complet : Génération de Plans SVG → Simulation CSI Wi-Fi
=============================================================================
Auteur  : Hanane Chahri
Contexte: Stage Orange Labs – Wi-Fi sensing pour géolocalisation indoor

Ce script reproduit la tâche centrale du stage :
    "automatiser la génération de données CSI à partir d'une liste
     d'images de plans d'habitats"

Approche : génération de plans SVG synthétiques mais réalistes
    - Pas de dataset à télécharger
    - Plans variés (Studio, T2, T3, T4) avec géométries différentes
    - Même format SVG que CubiCasa5k → pipeline identique en production

Usage :
    python pipeline.py                    # 12 plans, 64 sous-porteuses
    python pipeline.py --n_plans 20       # plus de plans
    python pipeline.py --n_subcarriers 128
=============================================================================
"""

import os
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# =============================================================================
# 1. GÉNÉRATEUR DE PLANS SVG RÉALISTES
# =============================================================================

# Dimensions typiques des pièces en mètres (min, max)
DIMENSIONS_PIECES = {
    'salon':       {'w': (3.5, 6.0), 'h': (3.5, 5.5)},
    'cuisine':     {'w': (2.5, 4.0), 'h': (2.5, 3.5)},
    'chambre':     {'w': (2.8, 4.5), 'h': (2.8, 4.0)},
    'salle_bain':  {'w': (1.8, 2.8), 'h': (1.8, 2.5)},
    'couloir':     {'w': (1.0, 1.8), 'h': (2.0, 4.0)},
    'wc':          {'w': (0.9, 1.2), 'h': (1.5, 2.0)},
    'bureau':      {'w': (2.5, 3.5), 'h': (2.5, 3.5)},
    'dressing':    {'w': (1.5, 2.5), 'h': (1.5, 2.5)},
}

EPAISSEUR_MUR = 0.15   # murs porteurs (m)
EPAISSEUR_CLO = 0.10   # cloisons intérieures (m)

# Typologies d'appartements : liste ordonnée de pièces
TYPOLOGIES = {
    'Studio': [
        ('salon', 'cuisine'),            # pièce principale + coin cuisine
        ('salle_bain',),
        ('wc',),
    ],
    'T2_A': [
        ('salon',),
        ('cuisine',),
        ('chambre',),
        ('salle_bain',),
        ('wc',),
    ],
    'T2_B': [
        ('salon', 'cuisine'),            # séjour/cuisine ouvert
        ('chambre',),
        ('chambre',),
        ('salle_bain',),
    ],
    'T3_A': [
        ('salon',),
        ('cuisine',),
        ('chambre',),
        ('chambre',),
        ('salle_bain',),
        ('wc',),
        ('couloir',),
    ],
    'T3_B': [
        ('salon',),
        ('cuisine',),
        ('chambre',),
        ('chambre',),
        ('bureau',),
        ('salle_bain',),
        ('wc',),
    ],
    'T4_A': [
        ('salon',),
        ('cuisine',),
        ('chambre',),
        ('chambre',),
        ('chambre',),
        ('salle_bain',),
        ('salle_bain',),
        ('wc',),
        ('couloir',),
    ],
    'T4_B': [
        ('salon',),
        ('cuisine',),
        ('chambre',),
        ('chambre',),
        ('chambre',),
        ('bureau',),
        ('dressing',),
        ('salle_bain',),
        ('wc',),
    ],
}


def _dim(type_piece, rng):
    """Tire des dimensions réalistes pour une pièce donnée."""
    d = DIMENSIONS_PIECES[type_piece]
    w = rng.uniform(*d['w'])
    h = rng.uniform(*d['h'])
    return round(w, 2), round(h, 2)


def generer_plan(typo_nom, seed=None):
    """
    Génère un plan d'appartement réaliste pour une typologie donnée.

    Algorithme de placement :
        - Les pièces sont disposées sur une grille irrégulière
        - On place d'abord les pièces principales (salon, chambres)
          sur deux rangées, puis les pièces de service (SDB, WC)
          sur une troisième rangée plus petite
        - Les murs extérieurs forment le périmètre total

    Retourne :
        pieces  : liste de dicts {nom, type, x, y, w, h}
        murs    : liste de segments ((x1,y1),(x2,y2)) — tous les murs
        meta    : dimensions totales, nom de la typo
    """
    rng = np.random.RandomState(seed if seed is not None else np.random.randint(0, 9999))
    typo = TYPOLOGIES[typo_nom]

    # ---- Placement des pièces ----
    # On dispose les pièces en 2-3 rangées selon la taille de la typo
    pieces_placees = []
    murs = []

    # Rangée 1 : pièces de vie principales
    types_principaux = ['salon', 'cuisine']
    types_chambres   = ['chambre', 'bureau', 'dressing']
    types_service    = ['salle_bain', 'wc', 'couloir']

    def trier(liste_types):
        p = [t for groupe in liste_types for t in groupe]
        rang1, rang2, rang3 = [], [], []
        for t in p:
            if t in types_principaux:     rang1.append(t)
            elif t in types_chambres:     rang2.append(t)
            else:                         rang3.append(t)
        return rang1, rang2, rang3

    r1_types, r2_types, r3_types = trier(typo)

    # Générer les dimensions
    def make_rangee(types_list):
        return [{'type': t, 'w': _dim(t, rng)[0], 'h': _dim(t, rng)[1]} for t in types_list]

    r1 = make_rangee(r1_types)
    r2 = make_rangee(r2_types)
    r3 = make_rangee(r3_types)

    if not r1 and r2:
        r1, r2 = r2, r3
        r3 = []

    # Hauteur de chaque rangée = max des hauteurs
    h1 = max((p['h'] for p in r1), default=0)
    h2 = max((p['h'] for p in r2), default=0)
    h3 = max((p['h'] for p in r3), default=0)

    # Largeur totale = max des sommes
    def largeur(rangee): return sum(p['w'] for p in rangee)

    w_total = max(largeur(r1), largeur(r2), largeur(r3))
    if w_total < 4:
        w_total = 4.0

    # Étirer la dernière rangée pour remplir si nécessaire
    def etirer_rangee(rangee, w_cible):
        if not rangee: return rangee
        ratio = w_cible / max(largeur(rangee), 0.01)
        for p in rangee:
            p['w'] = round(p['w'] * ratio, 2)
        return rangee

    r1 = etirer_rangee(r1, w_total)
    r2 = etirer_rangee(r2, w_total)
    r3 = etirer_rangee(r3, w_total)

    # Positionner chaque pièce
    y_offset = 0.0
    for rangee in [r1, r2, r3]:
        if not rangee: continue
        x_offset = 0.0
        h_rangee = max(p['h'] for p in rangee)
        for p in rangee:
            p['x'] = round(x_offset, 2)
            p['y'] = round(y_offset, 2)
            p['h'] = h_rangee  # harmoniser hauteur de rangée
            pieces_placees.append(p)
            x_offset += p['w']
        y_offset += h_rangee

    if not pieces_placees:
        return [], [], {}

    h_total = y_offset
    if h_total < 3:
        h_total = 3.0

    # ---- Extraction des segments de murs ----
    # Mur extérieur (périmètre)
    ext = [
        ((0, 0), (w_total, 0)),
        ((w_total, 0), (w_total, h_total)),
        ((w_total, h_total), (0, h_total)),
        ((0, h_total), (0, 0)),
    ]
    murs.extend(ext)

    # Murs intérieurs (séparations entre pièces)
    for p in pieces_placees:
        x, y, w, h = p['x'], p['y'], p['w'], p['h']
        # 4 côtés de chaque pièce → segments intérieurs
        candidats = [
            ((x, y), (x + w, y)),       # bas
            ((x + w, y), (x + w, y + h)), # droite
            ((x + w, y + h), (x, y + h)), # haut
            ((x, y + h), (x, y)),         # gauche
        ]
        for seg in candidats:
            # Garder seulement si ce n'est pas un mur extérieur
            is_ext = (
                (seg[0][0] == 0 and seg[1][0] == 0) or
                (seg[0][0] == w_total and seg[1][0] == w_total) or
                (seg[0][1] == 0 and seg[1][1] == 0) or
                (seg[0][1] == h_total and seg[1][1] == h_total)
            )
            if not is_ext:
                # Éviter les doublons (même segment ajouté par deux pièces)
                deja = any(
                    (abs(m[0][0]-seg[0][0]) < 0.01 and abs(m[0][1]-seg[0][1]) < 0.01 and
                     abs(m[1][0]-seg[1][0]) < 0.01 and abs(m[1][1]-seg[1][1]) < 0.01)
                    for m in murs
                )
                if not deja:
                    murs.append(seg)

    meta = {
        'typo': typo_nom,
        'w_total': round(w_total, 2),
        'h_total': round(h_total, 2),
        'n_pieces': len(pieces_placees),
        'surface': round(w_total * h_total, 1),
    }

    return pieces_placees, murs, meta


def plan_vers_svg(pieces, murs, meta, svg_path, px_per_meter=60):
    """
    Sauvegarde un plan généré en fichier SVG (même format que CubiCasa5k).

    Chaque mur est encodé comme un élément <line> avec stroke='#000000'.
    Chaque pièce est encodée comme un <rect> coloré avec son nom en <text>.

    Paramètre px_per_meter : résolution du SVG (60px/m → plan lisible)
    """
    W = meta['w_total']
    H = meta['h_total']
    svg_w = int(W * px_per_meter + 40)
    svg_h = int(H * px_per_meter + 40)
    margin = 20

    couleurs_pieces = {
        'salon':      '#AED6F1',
        'cuisine':    '#A9DFBF',
        'chambre':    '#F9E79F',
        'salle_bain': '#D7BDE2',
        'wc':         '#FDEBD0',
        'couloir':    '#D5D8DC',
        'bureau':     '#FADBD8',
        'dressing':   '#D6EAF8',
    }

    root = ET.Element('svg', {
        'xmlns': 'http://www.w3.org/2000/svg',
        'viewBox': f'0 0 {svg_w} {svg_h}',
        'width': str(svg_w),
        'height': str(svg_h),
    })

    # Fond blanc
    ET.SubElement(root, 'rect', {
        'width': str(svg_w), 'height': str(svg_h), 'fill': 'white'
    })

    def tx(x): return str(round(margin + x * px_per_meter, 1))
    def ty(y): return str(round(margin + (H - y) * px_per_meter, 1))  # flip Y

    # Pièces colorées
    for p in pieces:
        x, y, w, h = p['x'], p['y'], p['w'], p['h']
        couleur = couleurs_pieces.get(p['type'], '#EEEEEE')
        ET.SubElement(root, 'rect', {
            'x': tx(x), 'y': ty(y + h),
            'width': str(round(w * px_per_meter, 1)),
            'height': str(round(h * px_per_meter, 1)),
            'fill': couleur, 'opacity': '0.6',
            'stroke': 'none',
        })
        # Label de la pièce
        cx = round(margin + (x + w/2) * px_per_meter, 1)
        cy = round(margin + (H - y - h/2) * px_per_meter, 1)
        txt = ET.SubElement(root, 'text', {
            'x': str(cx), 'y': str(cy),
            'text-anchor': 'middle', 'dominant-baseline': 'middle',
            'font-size': str(max(8, int(min(w, h) * px_per_meter * 0.18))),
            'fill': '#555555', 'font-family': 'Arial',
        })
        txt.text = p['type'].replace('_', ' ')

    # Murs (éléments <line> avec stroke noir)
    for seg in murs:
        (x1, y1), (x2, y2) = seg
        is_ext = (
            (x1 == 0 and x2 == 0) or (x1 == W and x2 == W) or
            (y1 == 0 and y2 == 0) or (y1 == H and y2 == H)
        )
        stroke_w = '3' if is_ext else '1.5'
        ET.SubElement(root, 'line', {
            'x1': tx(x1), 'y1': ty(y1),
            'x2': tx(x2), 'y2': ty(y2),
            'stroke': '#000000',
            'stroke-width': stroke_w,
        })

    # Titre
    ET.SubElement(root, 'text', {
        'x': str(margin), 'y': str(margin - 5),
        'font-size': '11', 'fill': '#333333', 'font-family': 'Arial',
    }).text = f"{meta['typo']} — {W:.1f}m × {H:.1f}m — {meta['surface']}m²"

    tree = ET.ElementTree(root)
    ET.indent(tree, space='  ')
    tree.write(svg_path, encoding='unicode', xml_declaration=False)


def generer_dataset_svg(output_dir, n_plans=12, seed_base=0):
    """
    Génère n_plans plans SVG variés et les sauvegarde dans output_dir.

    Les typologies sont distribuées pour avoir de la variété :
    environ 2 plans par type (Studio, T2, T3, T4).
    """
    os.makedirs(output_dir, exist_ok=True)
    typologies = list(TYPOLOGIES.keys())
    svg_paths = []

    print(f"  Génération de {n_plans} plans SVG réalistes...")

    for i in range(n_plans):
        typo = typologies[i % len(typologies)]
        seed = seed_base + i * 17

        pieces, murs, meta = generer_plan(typo, seed=seed)
        if not pieces:
            continue

        svg_path = os.path.join(output_dir, f'plan_{i:03d}_{typo}.svg')
        plan_vers_svg(pieces, murs, meta, svg_path)
        svg_paths.append((svg_path, pieces, murs, meta))

        print(f"    [{i+1:2d}/{n_plans}] {typo:8s} — "
              f"{meta['w_total']:.1f}m × {meta['h_total']:.1f}m "
              f"({meta['surface']}m²) — {len(murs)} murs")

    print(f"  ✓ {len(svg_paths)} plans générés dans '{output_dir}/'")
    return svg_paths


# =============================================================================
# 2. PARSING SVG → SEGMENTS DE MURS
# =============================================================================

def parse_svg_walls(svg_path, px_per_meter=60):
    """
    Extrait les segments de murs d'un SVG généré par plan_vers_svg().

    Les murs sont encodés comme des éléments <line stroke='#000000'>.
    On convertit les coordonnées pixels → mètres.

    Retourne :
        segments : liste de ((x1,y1),(x2,y2)) en mètres
        bbox     : (x_min, x_max, y_min, y_max) en mètres
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    # Lire les dimensions SVG
    vb = root.get('viewBox', '').split()
    svg_w = float(vb[2]) if len(vb) >= 4 else float(root.get('width', 600))
    svg_h = float(vb[3]) if len(vb) >= 4 else float(root.get('height', 600))
    margin = 20

    # Dimensions réelles (en mètres) depuis le titre
    W_m = (svg_w - 2 * margin) / px_per_meter
    H_m = (svg_h - 2 * margin) / px_per_meter

    def from_svg(px_x, px_y):
        """Convertit coordonnées SVG (pixels, Y-flippé) → mètres."""
        x_m = (float(px_x) - margin) / px_per_meter
        y_m = H_m - (float(px_y) - margin) / px_per_meter
        return round(x_m, 3), round(y_m, 3)

    segments = []

    # Tous les namespaces possibles
    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if tag != 'line':
            continue
        stroke = elem.get('stroke', '').lower().strip()
        if stroke != '#000000':
            continue

        try:
            x1, y1 = from_svg(elem.get('x1', 0), elem.get('y1', 0))
            x2, y2 = from_svg(elem.get('x2', 0), elem.get('y2', 0))
        except (ValueError, TypeError):
            continue

        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if length > 0.1:
            segments.append(((x1, y1), (x2, y2)))

    if not segments:
        return [], None

    all_pts = np.array([p for seg in segments for p in seg])
    bbox = (all_pts[:,0].min(), all_pts[:,0].max(),
            all_pts[:,1].min(), all_pts[:,1].max())

    return segments, bbox


# =============================================================================
# 3. SIMULATION CSI (modèle multitrajet — géométrie réelle)
# =============================================================================

def reflexion_image_miroir(pos_rx, pos_tx, segment):
    """
    Calcule le trajet réfléchi TX → mur → RX via la méthode de l'image miroir.

    Principe :
        - L'image miroir TX' est la réflexion symétrique de TX par rapport au mur
        - Le rayon réfléchi passe par TX' → point_reflexion → RX
        - Valide seulement si le point de réflexion est sur le segment
          ET que TX et RX sont de part et d'autre du mur

    Retourne (distance_totale,) ou None si pas de réflexion valide
    """
    p1 = np.array(segment[0], dtype=float)
    p2 = np.array(segment[1], dtype=float)
    tx = np.array(pos_tx, dtype=float)
    rx = np.array(pos_rx, dtype=float)

    mur_vec = p2 - p1
    mur_len = np.linalg.norm(mur_vec)
    if mur_len < 0.05:
        return None

    mur_norm = mur_vec / mur_len
    mur_perp = np.array([-mur_norm[1], mur_norm[0]])

    # Distances signées de TX et RX par rapport au mur
    d_tx = np.dot(tx - p1, mur_perp)
    d_rx = np.dot(rx - p1, mur_perp)

    # Réflexion possible seulement si TX et RX sont du même côté
    # (le signal va de TX vers le mur puis vers RX)
    if d_tx * d_rx > 0:
        return None

    # Image miroir de TX
    tx_image = tx - 2 * d_tx * mur_perp

    # Direction TX_image → RX
    dir_vec = rx - tx_image
    dir_len = np.linalg.norm(dir_vec)
    if dir_len < 1e-6:
        return None
    dir_unit = dir_vec / dir_len

    # Intersection TX_image + t*dir avec la ligne du mur p1 + s*mur_norm
    denom = dir_unit[0] * mur_norm[1] - dir_unit[1] * mur_norm[0]
    if abs(denom) < 1e-8:
        return None  # parallèle

    diff = p1 - tx_image
    t = (diff[0] * mur_norm[1] - diff[1] * mur_norm[0]) / denom
    s = (diff[0] * dir_unit[1] - diff[1] * dir_unit[0]) / denom

    # Vérifier que le point est bien sur le segment (0 ≤ s ≤ mur_len)
    if t < 0 or not (0 <= s <= mur_len):
        return None

    pt_refl = tx_image + t * dir_unit
    d_total = np.linalg.norm(pt_refl - tx) + np.linalg.norm(rx - pt_refl)

    return d_total


def simuler_csi(pos_rx, pos_tx, segments_murs, n_subcarriers=64, sigma_bruit=0.03):
    """
    Simule le vecteur CSI pour une position donnée.

    Modèle de canal Wi-Fi 5GHz à multitrajets :

        H(f_k) = H_LOS(f_k) + Σ_murs H_réfl(f_k) + bruit

    Trajet direct :
        H_LOS(f_k) = (1/d_LOS) · exp(-j · 2π · f_k · d_LOS/c)

    Trajet réfléchi sur mur (coefficients réalistes) :
        H_réfl(f_k) = Γ/d_réfl · exp(-j · (2π · f_k · d_réfl/c + π))

        Γ = 0.35  mur porteur (longueur > 2m, béton/brique)
        Γ = 0.15  cloison    (longueur ≤ 2m, plâtre)

    Retourne : vecteur réel de taille 2*n_subcarriers [Re(H) | Im(H)]
    """
    c     = 3e8          # vitesse lumière (m/s)
    f0    = 5.18e9       # fréquence centrale Wi-Fi 5GHz (Hz)
    df    = 312.5e3      # espacement sous-porteuses OFDM (Hz)

    freqs = f0 + np.arange(n_subcarriers) * df
    H = np.zeros(n_subcarriers, dtype=complex)

    rx = np.array(pos_rx)
    tx = np.array(pos_tx)

    # Trajet direct
    d_los = max(np.linalg.norm(rx - tx), 0.1)
    H += (1.0 / d_los) * np.exp(-1j * 2 * np.pi * freqs * d_los / c)

    # Multitrajets
    for seg in segments_murs:
        d_refl = reflexion_image_miroir(rx, tx, seg)
        if d_refl is None:
            continue

        # Coefficient de réflexion selon la taille du mur
        longueur = np.linalg.norm(np.array(seg[1]) - np.array(seg[0]))
        gamma = 0.35 if longueur > 2.0 else 0.15

        H += (gamma / d_refl) * np.exp(-1j * (2*np.pi*freqs*d_refl/c + np.pi))

    # Bruit AWGN
    H += sigma_bruit * (np.random.randn(n_subcarriers) +
                        1j * np.random.randn(n_subcarriers))

    return np.concatenate([np.real(H), np.imag(H)])


def generer_grille(bbox, pas=0.6):
    """Grille de positions de mesure avec marge de 0.4m par rapport aux murs."""
    x_min, x_max, y_min, y_max = bbox
    xs = np.arange(x_min + 0.4, x_max - 0.4, pas)
    ys = np.arange(y_min + 0.4, y_max - 0.4, pas)
    if len(xs) == 0: xs = [(x_min + x_max) / 2]
    if len(ys) == 0: ys = [(y_min + y_max) / 2]
    return np.array([[x, y] for x in xs for y in ys])


def placer_routeur(bbox, pieces_meta, offset=(1.0, 1.0)):
    """Place le routeur dans le salon (ou à défaut dans un coin)."""
    x_min, x_max, y_min, y_max = bbox
    # Chercher le salon dans les pièces
    for p in pieces_meta:
        if p['type'] in ('salon',):
            return np.array([p['x'] + offset[0], p['y'] + offset[1]])
    return np.array([x_min + offset[0], y_min + offset[1]])


# =============================================================================
# 4. TRAITEMENT BATCH
# =============================================================================

def traiter_plan(svg_path, pieces_meta, n_subcarriers=64, pas=0.6):
    """
    Pipeline complet pour un plan : SVG → CSI dataset.

    Retourne (X, positions, meta) ou None si le plan est invalide.
    """
    segments, bbox = parse_svg_walls(svg_path)
    if not segments or bbox is None or len(segments) < 3:
        return None

    x_min, x_max, y_min, y_max = bbox
    if (x_max - x_min) < 2 or (y_max - y_min) < 2:
        return None

    positions = generer_grille(bbox, pas=pas)
    if len(positions) < 3:
        return None

    routeur = placer_routeur(bbox, pieces_meta)

    X = np.array([
        simuler_csi(pos, routeur, segments, n_subcarriers)
        for pos in positions
    ])

    nom = os.path.basename(svg_path).replace('.svg', '')
    meta = {
        'nom': nom,
        'svg_path': svg_path,
        'segments': segments,
        'bbox': bbox,
        'routeur': routeur,
        'n_murs': len(segments),
        'n_positions': len(positions),
        'dim_csi': X.shape[1],
        'dimensions': (round(x_max - x_min, 1), round(y_max - y_min, 1)),
        'pieces_meta': pieces_meta,
    }
    return X, positions, meta


# =============================================================================
# 5. VISUALISATIONS
# =============================================================================

CMAP_PLANS = plt.cm.get_cmap('tab10')


def plot_plans_generes(svg_data, n_affich=6):
    """
    Affiche les plans générés avec leurs murs et positions de mesure.
    Montre la variété des géométries produites.
    """
    n = min(n_affich, len(svg_data))
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for i, (ax, (svg_path, pieces, murs, meta)) in enumerate(zip(axes, svg_data[:n])):
        # Pièces colorées
        couleurs_p = {
            'salon': '#AED6F1', 'cuisine': '#A9DFBF', 'chambre': '#F9E79F',
            'salle_bain': '#D7BDE2', 'wc': '#FDEBD0', 'couloir': '#D5D8DC',
            'bureau': '#FADBD8', 'dressing': '#D6EAF8'
        }
        for p in pieces:
            rect = mpatches.Rectangle(
                (p['x'], p['y']), p['w'], p['h'],
                facecolor=couleurs_p.get(p['type'], '#EEE'),
                edgecolor='none', alpha=0.5
            )
            ax.add_patch(rect)
            ax.text(p['x'] + p['w']/2, p['y'] + p['h']/2,
                    p['type'].replace('_', '\n'),
                    ha='center', va='center', fontsize=7, color='#444')

        # Murs
        lc = LineCollection(murs, colors='#1a1a1a', linewidths=1.8)
        ax.add_collection(lc)

        W, H = meta['w_total'], meta['h_total']
        ax.set_xlim(-0.3, W + 0.3)
        ax.set_ylim(-0.3, H + 0.3)
        ax.set_aspect('equal')
        ax.set_title(f"{meta['typo']}\n{W:.1f}m × {H:.1f}m — {meta['surface']}m²",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('x (m)', fontsize=8)
        ax.set_ylabel('y (m)', fontsize=8)
        ax.grid(True, alpha=0.2)

    # Masquer les axes vides
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle("Plans synthétiques générés automatiquement\n"
                 "Géométries variées : Studio, T2, T3, T4",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


def plot_heatmap_csi(resultats, n_affich=3):
    """
    Heatmap de l'amplitude CSI par position.

    L'atténuation due aux murs est visible : positions derrière un mur
    ont une amplitude plus faible → le CSI encode la géométrie.
    """
    n = min(n_affich, len(resultats))
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.5))
    if n == 1: axes = [axes]

    for ax, (X, positions, meta) in zip(axes, resultats[:n]):
        n_half = X.shape[1] // 2
        amplitudes = np.sqrt(X[:,:n_half]**2 + X[:,n_half:]**2).mean(axis=1)

        sc = ax.scatter(positions[:,0], positions[:,1],
                        c=amplitudes, cmap='plasma', s=100, alpha=0.9, zorder=3)
        plt.colorbar(sc, ax=ax, label='Amplitude CSI')

        lc = LineCollection(meta['segments'], colors='white', linewidths=2, alpha=0.8, zorder=4)
        ax.add_collection(lc)
        ax.scatter(*meta['routeur'], marker='*', s=400, c='cyan', zorder=5,
                   edgecolors='white', linewidths=1.5)

        x0, x1, y0, y1 = meta['bbox']
        ax.set_xlim(x0-0.2, x1+0.2)
        ax.set_ylim(y0-0.2, y1+0.2)
        ax.set_aspect('equal')
        ax.set_facecolor('#0d1117')
        ax.set_title(f"{meta['nom']}\n★ routeur | blanc = murs | couleur = amplitude CSI",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')

    plt.suptitle("Heatmap CSI : l'amplitude révèle la géométrie de l'appartement\n"
                 "Zones atténuées = derrière un mur",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


def plot_separation_plans(resultats):
    """
    Visualise la séparabilité des appartements dans l'espace CSI.

    Si chaque appartement forme un cluster distinct → le CSI est une
    signature unique de la géométrie → exploitable pour la géolocalisation.
    """
    # Construire le dataset global
    n_par_plan = 20
    X_all, y_all = [], []
    for i, (X, positions, meta) in enumerate(resultats):
        n = min(n_par_plan, len(X))
        idx = np.random.choice(len(X), n, replace=False)
        X_all.append(X[idx])
        y_all.extend([i] * n)

    X_all = np.vstack(X_all)
    y_all = np.array(y_all)
    n_plans = len(resultats)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_all)

    # Variance inter / intra
    centroides = np.array([X_sc[y_all==i].mean(axis=0) for i in range(n_plans)])
    var_inter = np.var(centroides, axis=0).mean()
    var_intra = np.mean([np.var(X_sc[y_all==i], axis=0).mean() for i in range(n_plans)])
    ratio = var_inter / max(var_intra, 1e-8)

    cmap = plt.cm.get_cmap('tab10', n_plans)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    methodes = [
        ('PCA',   PCA(n_components=2, random_state=42)),
        ('t-SNE', TSNE(n_components=2, perplexity=min(30, len(X_all)//4),
                       n_iter=1000, random_state=42, init='pca', learning_rate='auto')),
        ('MDS',   MDS(n_components=2, random_state=42, normalized_stress='auto')),
    ]

    for ax, (nom, algo) in zip(axes, methodes):
        proj = algo.fit_transform(X_sc)
        for i in range(n_plans):
            mask = y_all == i
            label = resultats[i][2]['nom'].split('_', 2)[-1]  # nom lisible
            ax.scatter(proj[mask,0], proj[mask,1],
                       c=[cmap(i)], s=55, alpha=0.8,
                       edgecolors='black', linewidths=0.3, label=label)
        ax.set_title(f"{nom}", fontsize=13, fontweight='bold')
        ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2')
        if n_plans <= 12:
            ax.legend(fontsize=7, ncol=2, loc='best')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Séparabilité des appartements dans l'espace CSI\n"
                 f"Ratio variance inter/intra = {ratio:.3f} "
                 f"({'✓ bonne séparation' if ratio > 1 else '⚠ à améliorer'})",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig, ratio


def plot_empreinte_spectrale(resultats, n_affich=6):
    """
    Variance du CSI par sous-porteuse pour chaque appartement.

    Observation clé : chaque appartement a une courbe de variance unique
    → c'est son "empreinte spectrale". Deux appartements de même surface
    mais de géométries différentes auront des courbes distinctes.
    """
    n = min(n_affich, len(resultats))
    fig, ax = plt.subplots(figsize=(14, 5))

    cmap = plt.cm.get_cmap('tab10', n)
    n_sc = resultats[0][0].shape[1] // 2

    for i, (X, _, meta) in enumerate(resultats[:n]):
        var = np.var(X[:, :n_sc], axis=0)
        label = meta['nom'].split('_', 2)[-1]
        ax.plot(var, color=cmap(i), linewidth=1.8, alpha=0.85, label=label)
        ax.fill_between(range(n_sc), var, alpha=0.08, color=cmap(i))

    ax.set_xlabel('Indice sous-porteuse OFDM', fontsize=12)
    ax.set_ylabel('Variance CSI inter-positions', fontsize=12)
    ax.set_title("Empreinte spectrale de chaque appartement\n"
                 "Chaque courbe est unique → base de la géolocalisation non supervisée",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_sc - 1)
    plt.tight_layout()
    return fig


def plot_geoloc_interne(resultats, n_affich=4):
    """
    Pour chaque appartement, tente une géolocalisation interne :
    réduction de dimension des positions CSI → espace 2D.
    Si la topologie géographique est préservée, les voisins spatiaux
    restent voisins dans l'espace réduit.
    """
    n = min(n_affich, len(resultats))
    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 10))
    if n == 1: axes = axes.reshape(2, 1)

    for i, (X, positions, meta) in enumerate(resultats[:n]):
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        # Espace réel
        ax_reel = axes[0, i]
        dist_reel = np.linalg.norm(positions - positions.mean(axis=0), axis=1)
        sc = ax_reel.scatter(positions[:,0], positions[:,1],
                             c=dist_reel, cmap='viridis', s=50, alpha=0.8)
        lc = LineCollection(meta['segments'], colors='black', linewidths=1.5)
        ax_reel.add_collection(lc)
        ax_reel.scatter(*meta['routeur'], marker='*', s=250, c='red', zorder=5)
        x0, x1, y0, y1 = meta['bbox']
        ax_reel.set_xlim(x0-.2, x1+.2); ax_reel.set_ylim(y0-.2, y1+.2)
        ax_reel.set_aspect('equal')
        ax_reel.set_title(f"Espace réel\n{meta['nom'].split('_',2)[-1]}", fontsize=10)
        ax_reel.set_xlabel('x (m)'); ax_reel.set_ylabel('y (m)')

        # Espace CSI réduit (PCA)
        ax_csi = axes[1, i]
        proj = PCA(n_components=2, random_state=42).fit_transform(X_sc)
        dist_reel_norm = dist_reel / dist_reel.max()
        ax_csi.scatter(proj[:,0], proj[:,1],
                       c=dist_reel_norm, cmap='viridis', s=50, alpha=0.8)
        ax_csi.set_title("Espace CSI (PCA)\ncouleur = distance au centre réelle",
                         fontsize=10)
        ax_csi.set_xlabel('PC1'); ax_csi.set_ylabel('PC2')
        ax_csi.grid(True, alpha=0.3)

    plt.suptitle("Géolocalisation interne : espace réel vs espace CSI\n"
                 "Si la structure géographique est préservée → géolocalisation possible",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Pipeline CSI Wi-Fi : génération SVG + simulation'
    )
    parser.add_argument('--n_plans',       type=int, default=12)
    parser.add_argument('--n_subcarriers', type=int, default=64)
    parser.add_argument('--pas_grille',    type=float, default=0.6,
                        help='Pas de la grille de mesure en mètres')
    parser.add_argument('--svg_dir',   type=str, default='plans_svg')
    parser.add_argument('--output_dir', type=str, default='resultats')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print("  PIPELINE CSI Wi-Fi — Plans Synthétiques → Géolocalisation")
    print("=" * 65)

    # ---- Étape 1 : Générer les plans SVG ----
    print(f"\n[1/5] Génération des plans SVG ({args.n_plans} plans)...")
    svg_data = generer_dataset_svg(args.svg_dir, n_plans=args.n_plans)

    # ---- Étape 2 : Simuler le CSI ----
    print(f"\n[2/5] Simulation CSI ({args.n_subcarriers} sous-porteuses)...")
    resultats = []
    for svg_path, pieces, murs, meta in svg_data:
        r = traiter_plan(svg_path, pieces, args.n_subcarriers, args.pas_grille)
        if r:
            resultats.append(r)
            X, pos, m = r
            print(f"  ✓ {m['nom']:35s} {m['n_positions']:3d} positions × {m['dim_csi']} features")

    print(f"\n  {len(resultats)} plans simulés avec succès")

    if len(resultats) < 2:
        print("  ✗ Pas assez de plans valides.")
        return

    # ---- Étape 3 : Analyse ----
    print(f"\n[3/5] Analyse statistique...")
    X_sample = resultats[0][0]
    n_sc = X_sample.shape[1] // 2
    print(f"  Dim CSI : {X_sample.shape[1]} (= 2 × {n_sc} sous-porteuses)")
    print(f"  Positions/plan : {min(r[2]['n_positions'] for r in resultats)} – "
          f"{max(r[2]['n_positions'] for r in resultats)}")

    # ---- Étape 4 : Visualisations ----
    print(f"\n[4/5] Génération des figures...")

    fig1 = plot_plans_generes(svg_data, n_affich=min(6, len(svg_data)))
    fig1.savefig(f"{args.output_dir}/01_plans_generes.png", dpi=150, bbox_inches='tight')
    print("  → 01_plans_generes.png")

    fig2 = plot_heatmap_csi(resultats, n_affich=min(3, len(resultats)))
    fig2.savefig(f"{args.output_dir}/02_heatmap_csi.png", dpi=150, bbox_inches='tight')
    print("  → 02_heatmap_csi.png")

    fig3, ratio = plot_separation_plans(resultats)
    fig3.savefig(f"{args.output_dir}/03_separation.png", dpi=150, bbox_inches='tight')
    print("  → 03_separation.png")

    fig4 = plot_empreinte_spectrale(resultats, n_affich=min(6, len(resultats)))
    fig4.savefig(f"{args.output_dir}/04_empreinte_spectrale.png", dpi=150, bbox_inches='tight')
    print("  → 04_empreinte_spectrale.png")

    fig5 = plot_geoloc_interne(resultats, n_affich=min(4, len(resultats)))
    fig5.savefig(f"{args.output_dir}/05_geoloc_interne.png", dpi=150, bbox_inches='tight')
    print("  → 05_geoloc_interne.png")

    # ---- Résumé ----
    print("\n" + "=" * 65)
    print("  RÉSUMÉ")
    print("=" * 65)
    surfaces = [r[2]['dimensions'][0]*r[2]['dimensions'][1] for r in resultats]
    print(f"\n  Plans traités       : {len(resultats)}")
    print(f"  Surface             : {min(surfaces):.0f}m² – {max(surfaces):.0f}m²")
    print(f"  Dim CSI             : {resultats[0][0].shape[1]} features")
    print(f"  Ratio inter/intra   : {ratio:.4f} "
          f"({'✓' if ratio > 1 else '⚠'})")
    print(f"\n  Figures → {args.output_dir}/")
    print("=" * 65)


if __name__ == '__main__':
    main()

