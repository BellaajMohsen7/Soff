#!/usr/bin/env python3
"""
Sofiene - Expert en Belote Tunisienne Contrée (Enhanced Version)
Bot intelligent basé sur les règles officielles avec compréhension linguistique avancée
Développé par BellaajMohsen7 - 2025
"""

import streamlit as st
import numpy as np
import pickle
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from difflib import SequenceMatcher
import json

# Import required libraries with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from fuzzywuzzy import fuzz, process
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    st.error("Veuillez installer les dépendances: pip install sentence-transformers scikit-learn fuzzywuzzy python-levenshtein")

@dataclass
class RuleMatch:
    rule_id: str
    score: float
    rule_data: Dict
    match_type: str = "semantic"  # semantic, fuzzy, pattern, exact

@dataclass
class HandEvaluation:
    recommended_announcement: int
    confidence: float
    reasoning: str
    alternative_options: List[int]
    detailed_analysis: str = ""

class LanguageProcessor:
    """Processeur linguistique avancé pour Français et Anglais"""
    
    def __init__(self):
        self.french_synonyms = {
            'annonce': ['annonce', 'contrat', 'enchère', 'déclaration', 'offre', 'bid'],
            'règle': ['règle', 'regle', 'loi', 'norme', 'principe', 'rule'],
            'recommandation': ['recommandation', 'conseil', 'suggestion', 'avis', 'guide'],
            'calculer': ['calculer', 'compter', 'évaluer', 'mesurer', 'déterminer'],
            'score': ['score', 'point', 'résultat', 'total', 'comptage'],
            'belote': ['belote', 'rebelote', 'roi', 'dame', 'king', 'queen'],
            'atout': ['atout', 'trump', 'couleur', 'suite'],
            'capot': ['capot', 'tous', 'plis', 'tricks', 'all'],
            'coinche': ['coinche', 'surcoinche', 'multiplicateur', 'doubler']
        }
        
        self.english_synonyms = {
            'announce': ['announce', 'bid', 'contract', 'declare', 'call'],
            'rule': ['rule', 'law', 'regulation', 'principle', 'guideline'],
            'recommendation': ['recommendation', 'advice', 'suggestion', 'tip', 'guide'],
            'calculate': ['calculate', 'count', 'compute', 'evaluate', 'determine'],
            'score': ['score', 'points', 'result', 'total', 'count'],
            'belote': ['belote', 'rebelote', 'king', 'queen', 'roi', 'dame'],
            'trump': ['trump', 'atout', 'suit', 'color'],
            'capot': ['capot', 'all', 'tricks', 'tous', 'plis'],
            'coinche': ['coinche', 'surcoinche', 'multiplier', 'double']
        }
        
        # Patterns de variation commune
        self.common_variations = {
            'fr': {
                r'règle?\s*d[\'']?annonce?s?': 'règles annonces',
                r'comment\s+annoncer': 'comment annoncer',
                r'quand\s+annoncer': 'quand annoncer',
                r'que?\s+annoncer': 'que annoncer',
                r'calcul\s*(?:de\s*)?(?:score|point)s?': 'calcul score',
                r'belote\s*(?:et\s*)?rebelote': 'belote rebelote',
                r'roi\s*(?:et\s*)?dame': 'roi dame',
                r'multiplicateur|coinche': 'coinche',
                r'tous\s*(?:les\s*)?plis': 'capot'
            },
            'en': {
                r'announcement?\s*rules?': 'announcement rules',
                r'how\s+to\s+announce': 'how to announce',
                r'when\s+to\s+announce': 'when to announce',
                r'what\s+to\s+announce': 'what to announce',
                r'score?\s*calculation': 'score calculation',
                r'belote\s*(?:and\s*)?rebelote': 'belote rebelote',
                r'king\s*(?:and\s*)?queen': 'king queen',
                r'multiplier|coinche': 'coinche',
                r'all\s*tricks': 'capot'
            }
        }
    
    def normalize_query(self, query: str, language: str = 'fr') -> str:
        """Normaliser une requête"""
        query = query.lower().strip()
        
        # Appliquer les variations communes
        patterns = self.common_variations.get(language, {})
        for pattern, replacement in patterns.items():
            query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
        
        return query
    
    def extract_keywords(self, query: str, language: str = 'fr') -> Set[str]:
        """Extraire les mots-clés d'une requête"""
        normalized = self.normalize_query(query, language)
        words = re.findall(r'\b\w+\b', normalized)
        
        keywords = set(words)
        
        # Ajouter les synonymes
        synonyms_dict = self.french_synonyms if language == 'fr' else self.english_synonyms
        
        for word in words:
            for key, synonyms in synonyms_dict.items():
                if word in synonyms:
                    keywords.update(synonyms)
        
        return keywords
    
    def calculate_similarity(self, query1: str, query2: str) -> float:
        """Calculer la similarité entre deux requêtes"""
        return SequenceMatcher(None, query1.lower(), query2.lower()).ratio()

class EnhancedHandEvaluator:
    """Évaluateur de main expert amélioré"""
    
    def __init__(self):
        self.trump_values = {
            'valet': 20, 'v': 20, 'j': 20, 'jack': 20,
            '9': 14, 'neuf': 14, 'nine': 14,
            'as': 11, 'a': 11, 'ace': 11,
            '10': 10, 'dix': 10, 'ten': 10,
            'roi': 4, 'r': 4, 'k': 4, 'king': 4,
            'dame': 3, 'd': 3, 'q': 3, 'queen': 3,
            '8': 0, 'huit': 0, 'eight': 0,
            '7': 0, 'sept': 0, 'seven': 0
        }
        
        self.non_trump_values = {
            'as': 11, 'a': 11, 'ace': 11,
            '10': 10, 'dix': 10, 'ten': 10,
            'roi': 4, 'r': 4, 'k': 4, 'king': 4,
            'dame': 3, 'd': 3, 'q': 3, 'queen': 3,
            'valet': 2, 'v': 2, 'j': 2, 'jack': 2,
            '9': 0, 'neuf': 0, 'nine': 0,
            '8': 0, 'huit': 0, 'eight': 0,
            '7': 0, 'sept': 0, 'seven': 0
        }
    
    def evaluate_hand_advanced(self, description: str, language: str = 'fr') -> HandEvaluation:
        """Évaluation avancée avec analyse détaillée"""
        description_lower = description.lower()
        
        # Analyser les atouts
        trump_analysis = self._analyze_trumps(description_lower)
        color_analysis = self._analyze_colors(description_lower)
        
        # Déterminer la recommandation
        recommendation = self._determine_recommendation(trump_analysis, color_analysis, description_lower)
        
        # Analyse détaillée
        detailed_analysis = self._generate_detailed_analysis(trump_analysis, color_analysis, language)
        
        return HandEvaluation(
            recommended_announcement=recommendation['points'],
            confidence=recommendation['confidence'],
            reasoning=recommendation['reasoning'],
            alternative_options=recommendation['alternatives'],
            detailed_analysis=detailed_analysis
        )
    
    def _analyze_trumps(self, description: str) -> Dict:
        """Analyser les atouts dans la description"""
        trump_cards = []
        has_jack = any(word in description for word in ['valet', 'jack', 'v', 'j'])
        has_nine = any(word in description for word in ['9', 'neuf', 'nine'])
        has_ace = any(word in description for word in ['as', 'ace', 'a'])
        has_ten = any(word in description for word in ['10', 'dix', 'ten'])
        
        trump_count = len(re.findall(r'(?:valet|jack|9|neuf|as|ace|10|dix|roi|king|dame|queen)', description))
        
        return {
            'has_jack': has_jack,
            'has_nine': has_nine,
            'has_ace': has_ace,
            'has_ten': has_ten,
            'trump_count': trump_count,
            'complete_trumps': has_jack and has_nine and has_ace and has_ten
        }
    
    def _analyze_colors(self, description: str) -> Dict:
        """Analyser les couleurs dans la description"""
        colors = []
        if any(word in description for word in ['cœur', 'coeur', 'heart']):
            colors.append('heart')
        if any(word in description for word in ['carreau', 'diamond']):
            colors.append('diamond')
        if any(word in description for word in ['trèfle', 'trefle', 'club']):
            colors.append('club')
        if any(word in description for word in ['pique', 'spade']):
            colors.append('spade')
        
        return {
            'color_count': len(colors),
            'colors': colors
        }
    
    def _determine_recommendation(self, trump_analysis: Dict, color_analysis: Dict, description: str) -> Dict:
        """Déterminer la recommandation basée sur l'analyse"""
        
        # 140 points - Main exceptionnelle
        if trump_analysis['complete_trumps'] and trump_analysis['trump_count'] >= 6:
            return {
                'points': 140,
                'confidence': 0.85,
                'reasoning': "Main exceptionnelle détectée - adversaire aura maximum 1 pli",
                'alternatives': [130, 120]
            }
        
        # 130 points - 2 couleurs max + atouts complets
        if trump_analysis['complete_trumps'] and color_analysis['color_count'] <= 2:
            return {
                'points': 130,
                'confidence': 0.9,
                'reasoning': "Maximum 2 couleurs + atouts complets détectés",
                'alternatives': [120, 110]
            }
        
        # 120 points - 3 couleurs max + atouts complets
        if trump_analysis['complete_trumps'] and color_analysis['color_count'] <= 3:
            return {
                'points': 120,
                'confidence': 0.85,
                'reasoning': "Maximum 3 couleurs + atouts complets détectés",
                'alternatives': [110, 130]
            }
        
        # 110 points - Atouts complets
        if trump_analysis['complete_trumps']:
            return {
                'points': 110,
                'confidence': 0.9,
                'reasoning': "Atouts complets détectés (Valet, 9, As, 10)",
                'alternatives': [100, 120]
            }
        
        # 100 points - Flexibilité
        if trump_analysis['trump_count'] >= 3:
            return {
                'points': 100,
                'confidence': 0.7,
                'reasoning': "Main équilibrée - flexibilité maximale",
                'alternatives': [90, 110]
            }
        
        # 90 points - Configuration de base
        return {
            'points': 90,
            'confidence': 0.6,
            'reasoning': "Configuration de base recommandée avec 2 As minimum",
            'alternatives': [100]
        }
    
    def _generate_detailed_analysis(self, trump_analysis: Dict, color_analysis: Dict, language: str) -> str:
        """Générer une analyse détaillée"""
        if language == 'fr':
            analysis = f"""**Analyse détaillée de votre main:**

**Atouts détectés:**
• Valet: {'✅' if trump_analysis['has_jack'] else '❌'}
• 9: {'✅' if trump_analysis['has_nine'] else '❌'}
• As: {'✅' if trump_analysis['has_ace'] else '❌'}
• 10: {'✅' if trump_analysis['has_ten'] else '❌'}
• Atouts complets: {'✅' if trump_analysis['complete_trumps'] else '❌'}

**Distribution des couleurs:**
• Nombre de couleurs: {color_analysis['color_count']}
• Couleurs identifiées: {', '.join(color_analysis['colors']) if color_analysis['colors'] else 'Non spécifiées'}

**Recommandations stratégiques:**
• Conservez vos atouts pour les plis cruciaux
• Observez les cartes jouées par les adversaires
• Adaptez votre stratégie selon le contrat annoncé"""
        else:
            analysis = f"""**Detailed hand analysis:**

**Detected trumps:**
• Jack: {'✅' if trump_analysis['has_jack'] else '❌'}
• 9: {'✅' if trump_analysis['has_nine'] else '❌'}
• Ace: {'✅' if trump_analysis['has_ace'] else '❌'}
• 10: {'✅' if trump_analysis['has_ten'] else '❌'}
• Complete trumps: {'✅' if trump_analysis['complete_trumps'] else '❌'}

**Color distribution:**
• Number of colors: {color_analysis['color_count']}
• Identified colors: {', '.join(color_analysis['colors']) if color_analysis['colors'] else 'Not specified'}

**Strategic recommendations:**
• Keep your trumps for crucial tricks
• Observe cards played by opponents
• Adapt your strategy according to announced contract"""
        
        return analysis

class ComprehensiveRulesDatabase:
    """Base de données complète des règles de Belote Contrée"""
    
    def __init__(self):
        self.rules = self._initialize_comprehensive_rules()
        
    def _initialize_comprehensive_rules(self):
        """Initialiser la base complète des règles"""
        return {
            # Règles d'annonces complètes
            'announcement_rules_complete': {
                'id': 'announcement_rules_complete',
                'category': 'announcements',
                'title_fr': '📢 Règles Complètes des Annonces',
                'title_en': '📢 Complete Announcement Rules',
                'content_fr': """**Système complet des annonces officielles:**

**90 points:**
• **Critère officiel:** 2 As minimum
• Configuration de base acceptable
• Stratégie défensive recommandée

**100 points:**
• **Critère officiel:** "Généralement comme tu veux"
• Flexibilité maximale dans la composition
• Main équilibrée appréciée

**110 points:**
• **CRITÈRE OBLIGATOIRE:** Atouts Complets
• Être sûr de collecter toutes les cartes d'atout dès le début
• **Configuration requise:** (Valet, 9, As, 10) minimum
• **Alternative:** (Valet, 9, As, 2+ autres cartes d'atout)

**120 points:**
• **CRITÈRE STRICT:** Maximum 3 couleurs à la main + Atouts Complets
• Les 3 couleurs peuvent être: cœurs, trèfle, carreau (+ atout)
• **Cas particulier:** 6 cartes d'atout (dont Valet + 9) + 2 cartes de couleurs différentes

**130 points:**
• **CRITÈRE TRÈS STRICT:** Maximum 2 couleurs à la main + Atouts Complets
• **Cas particulier:** 6 cartes d'atout (dont Valet + 9) + 2 cartes même couleur ≠ atout

**140 points:**
• **CRITÈRE EXTRÊME:** L'adversaire ne peut avoir qu'un seul pli maximum
• Main quasi-parfaite obligatoire
• Risque très élevé""",
                'content_en': """**Complete official announcement system:**

**90 points:**
• **Official criterion:** Minimum 2 Aces
• Basic acceptable configuration
• Defensive strategy recommended

**100 points:**
• **Official criterion:** "Generally as you wish"
• Maximum flexibility in composition
• Balanced hand appreciated

**110 points:**
• **MANDATORY CRITERION:** Complete Trumps
• Must be sure to collect all trump cards from start
• **Required configuration:** (Jack, 9, Ace, 10) minimum
• **Alternative:** (Jack, 9, Ace, 2+ other trump cards)

**120 points:**
• **STRICT CRITERION:** Maximum 3 colors in hand + Complete Trumps
• The 3 colors can be: hearts, clubs, diamonds (+ trump)
• **Special case:** 6 trump cards (including Jack + 9) + 2 cards of different colors

**130 points:**
• **VERY STRICT CRITERION:** Maximum 2 colors in hand + Complete Trumps
• **Special case:** 6 trump cards (including Jack + 9) + 2 cards same color ≠ trump

**140 points:**
• **EXTREME CRITERION:** Opponent can have maximum one trick
• Near-perfect hand mandatory
• Very high risk""",
                'keywords_fr': ['annonce', 'règle', 'regle', 'recommandation', '90', '100', '110', '120', '130', '140', 'atouts', 'complets', 'couleurs', 'officiel', 'comment', 'quand', 'que'],
                'keywords_en': ['announcement', 'rule', 'recommendation', '90', '100', '110', '120', '130', '140', 'trumps', 'complete', 'colors', 'official', 'how', 'when', 'what'],
                'query_variations_fr': [
                    'règle annonce', 'regle annonce', 'règles annonces',
                    'comment annoncer', 'quand annoncer', 'que annoncer',
                    'recommandation annonce', 'critère annonce',
                    'annonce 90', 'annonce 100', 'annonce 110', 'annonce 120', 'annonce 130', 'annonce 140'
                ],
                'query_variations_en': [
                    'announcement rule', 'announce rule', 'bidding rule',
                    'how to announce', 'when to announce', 'what to announce',
                    'announcement recommendation', 'announcement criteria',
                    'announce 90', 'announce 100', 'announce 110', 'announce 120', 'announce 130', 'announce 140'
                ]
            },
            
            # Système de calcul complet
            'scoring_system_complete': {
                'id': 'scoring_system_complete',
                'category': 'scoring',
                'title_fr': '🔢 Système de Calcul Complet',
                'title_en': '🔢 Complete Scoring System',
                'content_fr': """**Système officiel de calcul des scores:**

**Points totaux possibles par manche:**
• Points des cartes: 152
• Dix de der (dernier pli): +10 points
• **Total possible: 162 points**

**Système de score spécial pour équipe non-preneuse:**
Si score = 10×K + x:
• Si x ∈ [5,6,7] → Score final = 10×(K+1)
• Sinon → Score final = 10×K
• Autre équipe: 160 - score calculé

**Belote/Rebelote:**
• +20 points si Roi et Dame d'atout chez même joueur
• Annonce obligatoire pour obtenir les points

**Échec de contrat:**
• Équipe preneuse: 0 points
• Équipe adverse: 160 + 20×(bonus belote)

**Capot (tous les plis):**
• 250 points automatiques
• Si dans contrat: DOIT faire tous les plis

**Coinche & Surcoinche:**
• Contrat simple: ×1
• Coinché: ×2
• Surcoinché: ×4

**Fin de partie:**
• Premier à 1001 points remporte
• Alternative: 2000 points selon accord""",
                'content_en': """**Official scoring system:**

**Total possible points per round:**
• Card points: 152
• Ten of last (last trick): +10 points
• **Total possible: 162 points**

**Special scoring system for non-taking team:**
If score = 10×K + x:
• If x ∈ [5,6,7] → Final score = 10×(K+1)
• Otherwise → Final score = 10×K
• Other team: 160 - calculated score

**Belote/Rebelote:**
• +20 points if King and Queen of trump with same player
• Announcement mandatory to get points

**Contract failure:**
• Taking team: 0 points
• Opposing team: 160 + 20×(belote bonus)

**Capot (all tricks):**
• 250 automatic points
• If in contract: MUST make all tricks

**Coinche & Surcoinche:**
• Simple contract: ×1
• Coinched: ×2
• Surcoinched: ×4

**Game end:**
• First to 1001 points wins
• Alternative: 2000 points by agreement""",
                'keywords_fr': ['score', 'calcul', 'points', 'système', 'comptage', 'total', 'belote', 'rebelote', 'capot', 'coinche', 'fin'],
                'keywords_en': ['score', 'calculation', 'points', 'system', 'counting', 'total', 'belote', 'rebelote', 'capot', 'coinche', 'end'],
                'query_variations_fr': [
                    'calcul score', 'calcul point', 'calculer points',
                    'système score', 'comptage', 'total points',
                    'comment compter', 'score final'
                ],
                'query_variations_en': [
                    'score calculation', 'point calculation', 'calculate points',
                    'scoring system', 'counting', 'total points',
                    'how to count', 'final score'
                ]
            },
            
            # Ajout des points au partenaire
            'partner_points_system': {
                'id': 'partner_points_system',
                'category': 'scoring',
                'title_fr': '🤝 Système d\'Ajout de Points au Partenaire',
                'title_en': '🤝 Partner Point Addition System',
                'content_fr': """**Système officiel d'ajout de points au partenaire:**

**Premier tour - Points d'atout:**
• **Avec Valet ou 9 d'atout:** (nombre de cartes d'atout - 1) × 10 points
• **Avec Valet seul:** +10 points
• **Sans Valet ni 9:** +10 points si 3 atouts minimum
• **Sinon:** Aucun ajout

**Deuxième tour - Points d'As:**
• **Ajout:** (nombre d'As × 10) points
• **Série consécutive commençant par As:** +20 points
  - Exemple: As-10-Roi = +20 points supplémentaires

**Troisième tour - Capot (très rare):**
• On cherche un capot potentiel
• **Ajout si:**
  - Vous avez des 10
  - Vous pouvez couper des couleurs avec vos atouts
• Évaluation situationnelle

**Exemples pratiques:**
• Main: Valet♠ 9♠ As♠ 7♠ + 4 autres → (4-1)×10 = 30 points
• Main: As♥ As♦ 10♥ → 2×10 = 20 points + série possible
• Main: As♣ 10♣ Roi♣ → 10 + 20 (série) = 30 points""",
                'content_en': """**Official partner point addition system:**

**First round - Trump points:**
• **With Jack or 9 of trump:** (number of trump cards - 1) × 10 points
• **With Jack alone:** +10 points
• **Without Jack or 9:** +10 points if 3+ trumps
• **Otherwise:** No addition

**Second round - Ace points:**
• **Addition:** (number of Aces × 10) points
• **Consecutive series starting with Ace:** +20 points
  - Example: Ace-10-King = +20 additional points

**Third round - Capot (very rare):**
• Looking for potential capot
• **Addition if:**
  - You have 10s
  - You can cut colors with your trumps
• Situational evaluation

**Practical examples:**
• Hand: Jack♠ 9♠ Ace♠ 7♠ + 4 others → (4-1)×10 = 30 points
• Hand: Ace♥ Ace♦ 10♥ → 2×10 = 20 points + possible series
• Hand: Ace♣ 10♣ King♣ → 10 + 20 (series) = 30 points""",
                'keywords_fr': ['partenaire', 'ajout', 'points', 'valet', 'as', 'série', 'atout', 'tour', 'calcul'],
                'keywords_en': ['partner', 'addition', 'points', 'jack', 'ace', 'series', 'trump', 'round', 'calculation'],
                'query_variations_fr': [
                    'ajout points partenaire', 'points partenaire', 'calcul partenaire',
                    'système partenaire', 'bonus partenaire'
                ],
                'query_variations_en': [
                    'partner points addition', 'partner points', 'partner calculation',
                    'partner system', 'partner bonus'
                ]
            },
            
            # Coinche et Surcoinche détaillé
            'coinche_system_detailed': {
                'id': 'coinche_system_detailed',
                'category': 'coinche',
                'title_fr': '🎯 Système Coinche & Surcoinche Détaillé',
                'title_en': '🎯 Detailed Coinche & Surcoinche System',
                'content_fr': """**Système officiel Coinche & Surcoinche:**

**Définitions:**
• **Coinche:** Doubler les enjeux d'un contrat adverse
• **Surcoinche:** Re-doubler après une coinche

**Multiplicateurs:**
• **Contrat simple:** ×1 (normal)
• **Contrat coinché:** ×2
• **Contrat surcoinché:** ×4

**Quand coincher:**
• Vous pensez que l'adversaire va chuter
• Votre main est forte contre leur annonce
• Vous avez des atouts dans leur couleur

**Risques et gains:**
• **Si adversaire chute:** Vous gagnez le double/quadruple
• **Si adversaire réussit:** Il gagne le double/quadruple

**Stratégie:**
• Coinchez uniquement si très confiant
• Attention aux contrats 90-100 (plus faciles)
• Évitez de coincher les mains exceptionnelles

**Exemples:**
• Contrat 110♠ coinché qui chute: 110×2 = 220 points
• Contrat 120♥ surcoinché réussi: 120×4 = 480 points

**Conseil d'expert:**
La coinche est une arme à double tranchant - utilisez-la avec parcimonie!""",
                'content_en': """**Official Coinche & Surcoinche system:**

**Definitions:**
• **Coinche:** Double the stakes of an opponent's contract
• **Surcoinche:** Re-double after a coinche

**Multipliers:**
• **Simple contract:** ×1 (normal)
• **Coinched contract:** ×2
• **Surcoinched contract:** ×4

**When to coinche:**
• You think opponent will fail
• Your hand is strong against their announcement
• You have trumps in their suit

**Risks and gains:**
• **If opponent fails:** You win double/quadruple
• **If opponent succeeds:** They win double/quadruple

**Strategy:**
• Only coinche if very confident
• Beware of 90-100 contracts (easier)
• Avoid coinching exceptional hands

**Examples:**
• 110♠ contract coinched that fails: 110×2 = 220 points
• 120♥ contract surcoinched that succeeds: 120×4 = 480 points

**Expert advice:**
Coinche is a double-edged sword - use it sparingly!""",
                'keywords_fr': ['coinche', 'surcoinche', 'multiplicateur', 'doubler', 'enjeux', 'stratégie', 'risque'],
                'keywords_en': ['coinche', 'surcoinche', 'multiplier', 'double', 'stakes', 'strategy', 'risk'],
                'query_variations_fr': [
                    'coinche surcoinche', 'multiplicateur', 'doubler contrat',
                    'quand coincher', 'stratégie coinche'
                ],
                'query_variations_en': [
                    'coinche surcoinche', 'multiplier', 'double contract',
                    'when to coinche', 'coinche strategy'
                ]
            },
            
            # Belote Rebelote détaillé
            'belote_rebelote_detailed': {
                'id': 'belote_rebelote_detailed',
                'category': 'bonus',
                'title_fr': '👑 Belote & Rebelote - Guide Complet',
                'title_en': '👑 Belote & Rebelote - Complete Guide',
                'content_fr': """**Guide complet Belote & Rebelote:**

**Définition officielle:**
• Avoir le Roi ET la Dame d'atout chez le même joueur
• Bonus: +20 points à l'équipe
• **Annonce OBLIGATOIRE** pour obtenir les points

**Procédure d'annonce:**
1. Annoncez "Belote" en jouant la première carte (Roi ou Dame)
2. Annoncez "Rebelote" en jouant la seconde carte
3. L'ordre Roi→Dame ou Dame→Roi n'importe pas

**Règles importantes:**
• Si oubli d'annoncer = PAS de bonus (0 points)
• Peut être joué à tout moment du jeu
• Valable uniquement si les deux cartes chez même joueur
• Ne peut pas être coinché/surcoinché

**Stratégies d'utilisation:**
• **Conservation:** Gardez pour moments cruciaux
• **Timing:** Jouez au bon moment pour remporter plis importants
• **Coordination:** Informez discrètement votre partenaire
• **Psychological:** Peut déstabiliser les adversaires

**Impact sur le score:**
• +20 points comptent dans le calcul final
• Peut faire la différence dans un contrat serré
• Compte même en cas de chute de contrat

**Exemples tactiques:**
• Utilisez pour prendre un pli de 10
• Gardez pour couper une couleur forte adverse
• Jouez en fin de partie pour sécuriser la victoire""",
                'content_en': """**Complete Belote & Rebelote guide:**

**Official definition:**
• Having King AND Queen of trump with same player
• Bonus: +20 points to the team
• **MANDATORY announcement** to get points

**Announcement procedure:**
1. Announce "Belote" when playing first card (King or Queen)
2. Announce "Rebelote" when playing second card
3. King→Queen or Queen→King order doesn't matter

**Important rules:**
• If forgotten to announce = NO bonus (0 points)
• Can be played anytime during game
• Valid only if both cards with same player
• Cannot be coinched/surcoinched

**Usage strategies:**
• **Conservation:** Keep for crucial moments
• **Timing:** Play at right time to win important tricks
• **Coordination:** Discretely inform your partner
• **Psychological:** Can destabilize opponents

**Score impact:**
• +20 points count in final calculation
• Can make difference in tight contract
• Counts even if contract fails

**Tactical examples:**
• Use to take a trick with 10
• Keep to cut strong opponent suit
• Play late game to secure victory""",
                'keywords_fr': ['belote', 'rebelote', 'roi', 'dame', 'atout', 'bonus', '20', 'points', 'annoncer', 'utiliser', 'stratégie'],
                'keywords_en': ['belote', 'rebelote', 'king', 'queen', 'trump', 'bonus', '20', 'points', 'announce', 'use', 'strategy'],
                'query_variations_fr': [
                    'belote rebelote', 'roi dame atout', 'bonus 20 points',
                    'quand utiliser belote', 'comment belote', 'stratégie belote'
                ],
                'query_variations_en': [
                    'belote rebelote', 'king queen trump', 'bonus 20 points',
                    'when use belote', 'how belote', 'belote strategy'
                ]
            },
            
            # Règles du Capot
            'capot_rules_complete': {
                'id': 'capot_rules_complete',
                'category': 'capot',
                'title_fr': '🏆 Règles Complètes du Capot',
                'title_en': '🏆 Complete Capot Rules',
                'content_fr': """**Règles officielles du Capot:**

**Définition:**
• Faire TOUS les plis (8 plis sur 8)
• Score automatique: 250 points
• Remplace le calcul normal des points

**Types de Capot:**

**1. Capot dans le contrat:**
• Annonce explicite: "Capot Cœur"
• **OBLIGATION:** Doit faire TOUS les plis
• Si échoue (même 7 plis sur 8): Chute totale
• Si réussit: 250 points

**2. Capot surprise:**
• Non annoncé mais réalisé
• Remplace automatiquement le contrat initial
• 250 points garantis

**Stratégies pour le Capot:**

**Conditions favorables:**
• Main exceptionnelle avec nombreux atouts
• Contrôle de plusieurs couleurs
• Partenaire fort probable

**Risques:**
• Très difficile à réaliser
• Un seul pli perdu = échec total
• Adversaires vont tout tenter pour prendre 1 pli

**Défense contre le Capot:**
• Conservez vos cartes fortes
• Tentez de prendre au moins 1 pli
• Coordination défensive avec partenaire

**Exemples de mains à Capot:**
• 6-7 atouts forts + As/10 dans autres couleurs
• Contrôle total d'une couleur + atouts complets
• Main quasi-parfaite avec domination évidente

**Conseil d'expert:**
Le Capot est spectaculaire mais très risqué - n'annoncez que si quasi-certain!""",
                'content_en': """**Official Capot rules:**

**Definition:**
• Make ALL tricks (8 out of 8)
• Automatic score: 250 points
• Replaces normal point calculation

**Types of Capot:**

**1. Capot in contract:**
• Explicit announcement: "Capot Hearts"
• **OBLIGATION:** Must make ALL tricks
• If fails (even 7 out of 8): Total failure
• If succeeds: 250 points

**2. Surprise Capot:**
• Not announced but achieved
• Automatically replaces initial contract
• 250 guaranteed points

**Capot strategies:**

**Favorable conditions:**
• Exceptional hand with many trumps
• Control of several suits
• Probably strong partner

**Risks:**
• Very difficult to achieve
• One lost trick = total failure
• Opponents will try everything for 1 trick

**Defense against Capot:**
• Keep your strong cards
• Try to take at least 1 trick
• Defensive coordination with partner

**Capot hand examples:**
• 6-7 strong trumps + Ace/10 in other suits
• Total control of one suit + complete trumps
• Near-perfect hand with obvious domination

**Expert advice:**
Capot is spectacular but very risky - only announce if almost certain!""",
                'keywords_fr': ['capot', 'tous', 'plis', '250', 'points', 'risque', 'stratégie', 'annoncer'],
                'keywords_en': ['capot', 'all', 'tricks', '250', 'points', 'risk', 'strategy', 'announce'],
                'query_variations_fr': [
                    'capot', 'tous les plis', '250 points', 'règles capot',
                    'quand capot', 'stratégie capot', 'risque capot'
                ],
                'query_variations_en': [
                    'capot', 'all tricks', '250 points', 'capot rules',
                    'when capot', 'capot strategy', 'capot risk'
                ]
            }
        }
    
    def get_all_rules(self):
        """Retourner toutes les règles"""
        return self.rules

class FuzzyMatcher:
    """Matcher flou pour gérer les variations et typos"""
    
    def __init__(self):
        self.min_similarity = 0.6
        self.exact_match_bonus = 0.3
        
    def find_best_matches(self, query: str, candidates: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """Trouver les meilleures correspondances floues"""
        if not DEPENDENCIES_AVAILABLE:
            return [(query, 1.0)]
            
        try:
            # Utiliser fuzzywuzzy pour le matching
            matches = process.extract(query, candidates, limit=top_k, scorer=fuzz.token_sort_ratio)
            
            # Convertir en format standard
            results = []
            for match, score in matches:
                normalized_score = score / 100.0
                if normalized_score >= self.min_similarity:
                    results.append((match, normalized_score))
            
            return results
        except Exception:
            # Fallback vers matching simple
            results = []
            query_lower = query.lower()
            
            for candidate in candidates:
                similarity = SequenceMatcher(None, query_lower, candidate.lower()).ratio()
                if similarity >= self.min_similarity:
                    results.append((candidate, similarity))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

class EnhancedSofieneAI:
    """Sofiene AI amélioré avec compréhension linguistique avancée"""
    
    def __init__(self):
        self.model = load_sentence_transformer()
        self.rules_db = ComprehensiveRulesDatabase()
        self.hand_evaluator = EnhancedHandEvaluator()
        self.language_processor = LanguageProcessor()
        self.fuzzy_matcher = FuzzyMatcher()
        self.rule_embeddings = {}
        self.context_window = 5
        
        # Cache pour améliorer les performances
        self.query_cache = {}
        self.max_cache_size = 100
        
        if self.model:
            self.initialize_embeddings()
    
    @st.cache_data
    def initialize_embeddings(_self):
        """Initialiser les embeddings avec cache"""
        embeddings_file = 'sofiene_enhanced_embeddings.pkl'
        
        if os.path.exists(embeddings_file):
            try:
                with open(embeddings_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return _self.compute_embeddings()
        else:
            return _self.compute_embeddings()
    
    def compute_embeddings(self):
        """Calculer les embeddings pour toutes les règles"""
        if not self.model:
            return {}
            
        try:
            embeddings = {}
            with st.spinner("Initialisation de l'expertise Sofiene améliorée..."):
                progress_bar = st.progress(0)
                rules = self.rules_db.get_all_rules()
                total_rules = len(rules)
                
                for i, (rule_id, rule) in enumerate(rules.items()):
                    # Texte français enrichi
                    text_fr = f"{rule['title_fr']} {rule['content_fr']} {' '.join(rule['keywords_fr'])}"
                    if 'query_variations_fr' in rule:
                        text_fr += f" {' '.join(rule['query_variations_fr'])}"
                    
                    # Texte anglais enrichi
                    text_en = f"{rule['title_en']} {rule['content_en']} {' '.join(rule['keywords_en'])}"
                    if 'query_variations_en' in rule:
                        text_en += f" {' '.join(rule['query_variations_en'])}"
                    
                    embedding_fr = self.model.encode(text_fr)
                    embedding_en = self.model.encode(text_en)
                    
                    embeddings[rule_id] = {
                        'fr': embedding_fr,
                        'en': embedding_en,
                        'rule': rule
                    }
                    
                    progress_bar.progress((i + 1) / total_rules)
                
                progress_bar.empty()
                
                # Sauvegarder les embeddings
                try:
                    with open('sofiene_enhanced_embeddings.pkl', 'wb') as f:
                        pickle.dump(embeddings, f)
                except Exception:
                    pass
                    
            return embeddings
        except Exception as e:
            st.error(f"Erreur de traitement: {str(e)}")
            return {}
    
    def process_query_enhanced(self, query: str, language: str = 'fr', context: List[str] = None) -> str:
        """Traitement de requête amélioré avec cache et fallbacks multiples"""
        
        # Vérifier le cache
        cache_key = f"{query}_{language}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        # Normaliser la requête
        normalized_query = self.language_processor.normalize_query(query, language)
        
        # Essayer différentes approches dans l'ordre
        response = None
        
        # 1. Patterns spécifiques améliorés
        response = self.handle_enhanced_patterns(query, language)
        if response:
            self._cache_response(cache_key, response)
            return response
        
        # 2. Recherche sémantique
        if self.model and self.rule_embeddings:
            response = self.semantic_search_enhanced(normalized_query, language)
            if response:
                self._cache_response(cache_key, response)
                return response
        
        # 3. Matching flou
        response = self.fuzzy_search(normalized_query, language)
        if response:
            self._cache_response(cache_key, response)
            return response
        
        # 4. Fallback intelligent
        response = self.intelligent_fallback(query, language, context)
        self._cache_response(cache_key, response)
        return response
    
    def handle_enhanced_patterns(self, query: str, language: str = 'fr') -> Optional[str]:
        """Gestion améliorée des patterns spécifiques"""
        query_lower = query.lower().strip()
        
        # Patterns d'évaluation de main améliorés
        hand_patterns = {
            'fr': [
                r'j.ai.*(?:valet|9|as|10|roi|dame).*(?:annoncer|conseiller)',
                r'(?:main|cartes?).*(?:annoncer|recommandation)',
                r'(?:que|quoi|combien).*annoncer.*(?:avec|main)',
                r'évaluer.*main', r'analyser.*main'
            ],
            'en': [
                r'i.have.*(?:jack|9|ace|10|king|queen).*(?:announce|recommend)',
                r'(?:hand|cards?).*(?:announce|recommendation)',
                r'(?:what|how much).*announce.*(?:with|hand)',
                r'evaluate.*hand', r'analyze.*hand'
            ]
        }
        
        patterns = hand_patterns.get(language, hand_patterns['fr'])
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return self.handle_hand_evaluation_enhanced(query, language)
        
        # Patterns d'annonces avec extraction de points
        points_extracted = self.extract_points_from_query(query_lower)
        if points_extracted:
            for points in points_extracted:
                if 90 <= points <= 140:
                    if any(word in query_lower for word in ['recommandation', 'recommendation', 'conseil', 'advice']):
                        return self.get_announcement_recommendation_enhanced(points, language)
                    elif any(word in query_lower for word in ['quand', 'when', 'comment', 'how']):
                        return self.get_announcement_conditions_enhanced(points, language)
        
        # Patterns Belote/Rebelote améliorés
        belote_patterns = {
            'fr': [
                r'belote.*rebelote', r'roi.*dame.*atout', r'bonus.*20',
                r'(?:quand|comment).*(?:utiliser|jouer).*belote',
                r'stratégie.*belote', r'belote.*stratégie'
            ],
            'en': [
                r'belote.*rebelote', r'king.*queen.*trump', r'bonus.*20',
                r'(?:when|how).*(?:use|play).*belote',
                r'strategy.*belote', r'belote.*strategy'
            ]
        }
        
        patterns = belote_patterns.get(language, belote_patterns['fr'])
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return self.get_belote_detailed_info(language)
        
        # Patterns Coinche/Surcoinche
        coinche_patterns = {
            'fr': [
                r'coinche.*surcoinche', r'multiplicateur', r'doubler.*contrat',
                r'(?:quand|comment).*coincher', r'stratégie.*coinche'
            ],
            'en': [
                r'coinche.*surcoinche', r'multiplier', r'double.*contract',
                r'(?:when|how).*coinche', r'strategy.*coinche'
            ]
        }
        
        patterns = coinche_patterns.get(language, coinche_patterns['fr'])
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return self.get_coinche_detailed_info(language)
        
        # Patterns Capot
        capot_patterns = {
            'fr': [
                r'capot', r'tous.*plis', r'250.*points',
                r'(?:quand|comment).*capot', r'stratégie.*capot'
            ],
            'en': [
                r'capot', r'all.*tricks', r'250.*points',
                r'(?:when|how).*capot', r'strategy.*capot'
            ]
        }
        
        patterns = capot_patterns.get(language, capot_patterns['fr'])
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return self.get_capot_detailed_info(language)
        
        return None
    
    def extract_points_from_query(self, query: str) -> List[int]:
        """Extraire les points mentionnés dans une requête"""
        points = []
        # Chercher les nombres entre 90 et 140
        matches = re.findall(r'\b(90|100|110|120|130|140)\b', query)
        for match in matches:
            points.append(int(match))
        return points
    
    def semantic_search_enhanced(self, query: str, language: str = 'fr') -> Optional[str]:
        """Recherche sémantique améliorée"""
        try:
            query_embedding = self.model.encode(query)
            matches = []
            
            # Extraire les mots-clés de la requête
            query_keywords = self.language_processor.extract_keywords(query, language)
            
            for rule_id, rule_data in self.rule_embeddings.items():
                rule_embedding = rule_data[language]
                rule = rule_data['rule']
                
                # Similarité sémantique
                similarity = cosine_similarity([query_embedding], [rule_embedding])[0][0]
                
                # Boost basé sur les mots-clés
                keyword_boost = self.calculate_keyword_boost(query_keywords, rule, language)
                similarity += keyword_boost
                
                # Boost basé sur les variations de requête
                variation_boost = self.calculate_variation_boost(query, rule, language)
                similarity += variation_boost
                
                matches.append(RuleMatch(
                    rule_id=rule_id,
                    score=similarity,
                    rule_data=rule,
                    match_type="semantic"
                ))
            
            matches.sort(key=lambda x: x.score, reverse=True)
            
            if matches and matches[0].score > 0.3:
                return self.generate_enhanced_response(matches[:3], query, language)
                
        except Exception as e:
            st.warning(f"Erreur de recherche sémantique: {str(e)}")
        
        return None
    
    def calculate_keyword_boost(self, query_keywords: Set[str], rule: Dict, language: str) -> float:
        """Calculer le boost basé sur les mots-clés"""
        rule_keywords = set(rule.get(f'keywords_{language}', []))
        
        # Intersection des mots-clés
        common_keywords = query_keywords.intersection(rule_keywords)
        
        if not rule_keywords:
            return 0
        
        # Score basé sur le pourcentage de mots-clés communs
        keyword_score = len(common_keywords) / len(rule_keywords)
        
        # Boost maximal de 0.4
        return min(keyword_score * 0.4, 0.4)
    
    def calculate_variation_boost(self, query: str, rule: Dict, language: str) -> float:
        """Calculer le boost basé sur les variations de requête"""
        query_lower = query.lower()
        variations = rule.get(f'query_variations_{language}', [])
        
        max_boost = 0
        for variation in variations:
            similarity = self.language_processor.calculate_similarity(query_lower, variation.lower())
            if similarity > 0.7:  # Seuil de similarité
                max_boost = max(max_boost, similarity * 0.3)
        
        return max_boost
    
    def fuzzy_search(self, query: str, language: str = 'fr') -> Optional[str]:
        """Recherche floue comme fallback"""
        try:
            # Construire la liste des variations de requête
            all_variations = []
            for rule_id, rule_data in self.rule_embeddings.items():
                rule = rule_data['rule']
                variations = rule.get(f'query_variations_{language}', [])
                for variation in variations:
                    all_variations.append((variation, rule_id, rule))
            
            # Chercher les meilleures correspondances floues
            variation_texts = [v[0] for v in all_variations]
            fuzzy_matches = self.fuzzy_matcher.find_best_matches(query, variation_texts, top_k=3)
            
            if fuzzy_matches and fuzzy_matches[0][1] > 0.7:
                # Trouver la règle correspondante
                best_variation = fuzzy_matches[0][0]
                for variation, rule_id, rule in all_variations:
                    if variation == best_variation:
                        match = RuleMatch(
                            rule_id=rule_id,
                            score=fuzzy_matches[0][1],
                            rule_data=rule,
                            match_type="fuzzy"
                        )
                        return self.generate_enhanced_response([match], query, language)
            
        except Exception as e:
            st.warning(f"Erreur de recherche floue: {str(e)}")
        
        return None
    
    def intelligent_fallback(self, query: str, language: str = 'fr', context: List[str] = None) -> str:
        """Fallback intelligent basé sur l'intention"""
        intent = self.extract_intent_enhanced(query, language)
        
        fallbacks = {
            'fr': {
                'announcement_rules': """Je peux vous expliquer les règles d'annonces complètes:

**Recommandations officielles:**
• **90 points:** 2 As minimum
• **100 points:** "Généralement comme tu veux"
• **110 points:** Atouts complets obligatoires
• **120 points:** Max 3 couleurs + atouts complets
• **130 points:** Max 2 couleurs + atouts complets
• **140 points:** Adversaire max 1 pli

Précisez votre question pour une réponse plus détaillée!""",
                
                'hand_evaluation': """Pour évaluer votre main, décrivez-moi vos cartes précisément:

**Format recommandé:**
"J'ai Valet, 9, As de carreau, plus 10 de cœur, Roi de trèfle..."

**Je peux analyser:**
• Votre potentiel d'annonce
• Les risques et opportunités
• La stratégie optimale
• Les alternatives possibles

Décrivez votre main et je vous donnerai une analyse experte!""",
                
                'scoring': """Le système de score de la Belote Contrée suit des règles précises:

**Points par manche:** 162 total (152 cartes + 10 dix de der)
**Belote/Rebelote:** +20 points
**Capot:** 250 points automatiques
**Coinche:** ×2, Surcoinche: ×4

Que souhaitez-vous savoir exactement sur le calcul des scores?""",
                
                'general': """Je suis Sofiene, votre expert en Belote Tunisienne Contrée amélioré!

**Mes spécialités:**
🎯 Règles d'annonces complètes (90-140 points)
🔍 Évaluation de main experte
📊 Calcul de scores et stratégies
👑 Belote/Rebelote et bonus
🏆 Capot et situations spéciales
🎲 Coinche/Surcoinche

Posez-moi une question précise et je vous donnerai une réponse experte!"""
            },
            'en': {
                'announcement_rules': """I can explain complete announcement rules:

**Official recommendations:**
• **90 points:** Minimum 2 Aces
• **100 points:** "Generally as you wish"
• **110 points:** Complete trumps mandatory
• **120 points:** Max 3 colors + complete trumps
• **130 points:** Max 2 colors + complete trumps
• **140 points:** Opponent max 1 trick

Please specify your question for a more detailed answer!""",
                
                'hand_evaluation': """To evaluate your hand, describe your cards precisely:

**Recommended format:**
"I have Jack, 9, Ace of diamonds, plus 10 of hearts, King of clubs..."

**I can analyze:**
• Your announcement potential
• Risks and opportunities
• Optimal strategy
• Possible alternatives

Describe your hand and I'll give you expert analysis!""",
                
                'scoring': """Belote Contrée scoring follows precise rules:

**Points per round:** 162 total (152 cards + 10 ten of last)
**Belote/Rebelote:** +20 points
**Capot:** 250 automatic points
**Coinche:** ×2, Surcoinche: ×4

What exactly would you like to know about score calculation?""",
                
                'general': """I'm Sofiene, your enhanced Tunisian Belote Contrée expert!

**My specialties:**
🎯 Complete announcement rules (90-140 points)
🔍 Expert hand evaluation
📊 Score calculation and strategies
👑 Belote/Rebelote and bonuses
🏆 Capot and special situations
🎲 Coinche/Surcoinche

Ask me a specific question and I'll give you an expert answer!"""
            }
        }
        
        return fallbacks.get(language, fallbacks['fr']).get(intent, fallbacks[language]['general'])
    
    def extract_intent_enhanced(self, query: str, language: str = 'fr') -> str:
        """Extraction d'intention améliorée"""
        query_lower = query.lower()
        keywords = self.language_processor.extract_keywords(query, language)
        
        # Priorités d'intention
        if any(word in keywords for word in ['belote', 'rebelote', 'roi', 'dame', 'king', 'queen']):
            return 'belote_rebelote'
        
        if any(word in keywords for word in ['coinche', 'surcoinche', 'multiplicateur', 'multiplier']):
            return 'coinche'
        
        if any(word in keywords for word in ['capot', 'tous', 'plis', 'all', 'tricks']):
            return 'capot'
        
        if any(word in keywords for word in ['main', 'hand', 'évaluer', 'evaluate', 'analyser', 'analyze']):
            return 'hand_evaluation'
        
        if any(word in keywords for word in ['annonce', 'announcement', 'recommandation', 'recommendation']):
            return 'announcement_rules'
        
        if any(word in keywords for word in ['score', 'calcul', 'calculation', 'points']):
            return 'scoring'
        
        return 'general'
    
    def handle_hand_evaluation_enhanced(self, query: str, language: str = 'fr') -> str:
        """Évaluation de main améliorée"""
        evaluation = self.hand_evaluator.evaluate_hand_advanced(query, language)
        
        if language == 'fr':
            response = f"""**🎯 Analyse experte de votre main par Sofiene**

**Recommandation officielle:** {evaluation.recommended_announcement} points
**Niveau de confiance:** {evaluation.confidence:.0%}

**Raisonnement:**
{evaluation.reasoning}

{evaluation.detailed_analysis}

**Alternatives envisageables:** {', '.join(map(str, evaluation.alternative_options))} points

**💡 Conseil d'expert Sofiene:**
Vérifiez que votre main respecte strictement les critères officiels avant d'annoncer. En cas de doute, optez pour une annonce plus conservatrice."""
        else:
            response = f"""**🎯 Sofiene's expert hand analysis**

**Official recommendation:** {evaluation.recommended_announcement} points
**Confidence level:** {evaluation.confidence:.0%}

**Reasoning:**
{evaluation.reasoning}

{evaluation.detailed_analysis}


**Possible alternatives:** {', '.join(map(str, evaluation.alternative_options))} points

**💡 Sofiene's expert advice:**
Verify your hand strictly meets official criteria before announcing. When in doubt, choose a more conservative announcement."""
        
        return response
    
    def get_announcement_recommendation_enhanced(self, points: int, language: str = 'fr') -> str:
        """Recommandations d'annonces améliorées avec exemples"""
        recommendations = {
            'fr': {
                90: """**📢 Recommandation officielle pour 90 points**

**Critère obligatoire:** 2 As minimum

**Configuration détaillée:**
• Main relativement faible mais jouable
• Au moins 2 As dans votre jeu (n'importe quelle couleur)
• Stratégie défensive acceptable
• Risque modéré

**Exemples de mains conformes:**
• As♠ As♥ + 6 autres cartes diverses
• As♦ As♣ + cartes moyennes
• As♠ As♦ + quelques figures

**💡 Conseil Sofiene:**
Annonce sûre et recommandée pour débuter. Idéale quand vous n'êtes pas sûr de votre main.""",
                
                100: """**📢 Recommandation officielle pour 100 points**

**Critère officiel:** "Généralement comme tu veux"

**Configuration détaillée:**
• Flexibilité maximale dans la composition
• Main équilibrée recommandée
• Quelques atouts appréciés mais non obligatoires
• Liberté totale de choix

**Exemples de mains conformes:**
• Composition libre avec bon équilibre
• Mix d'atouts et de cartes fortes
• Main sans critère strict

**💡 Conseil Sofiene:**
Annonce flexible parfaite pour s'adapter au jeu. Utilisez votre expérience pour juger.""",
                
                110: """**📢 Recommandation officielle pour 110 points**

**CRITÈRE OBLIGATOIRE:** Atouts Complets

**Configuration strictement requise:**
• Être sûr de collecter toutes les cartes d'atout dès le début
• **Option 1:** (Valet, 9, As, 10) d'atout minimum
• **Option 2:** (Valet, 9, As + 2+ autres cartes d'atout)
• Confiance totale dans le contrôle des atouts

**Exemples de mains conformes:**
• Valet♠ 9♠ As♠ 10♠ + 4 autres cartes
• Valet♥ 9♥ As♥ Roi♥ Dame♥ + 3 autres
• Valet♦ 9♦ As♦ 10♦ 8♦ 7♦ + 2 autres

**⚠️ ATTENTION:** Sans atouts complets, échec quasi-certain!

**💡 Conseil Sofiene:**
Ne prenez ce risque que si vous êtes absolument certain de contrôler tous les atouts.""",
                
                120: """**📢 Recommandation officielle pour 120 points**

**CRITÈRE OBLIGATOIRE:** Maximum 3 couleurs + Atouts Complets

**Configuration strictement requise:**
• Seulement 3 couleurs dans votre main (parmi: cœurs, trèfle, carreau, pique)
• Plus atouts complets d'une de ces couleurs
• Distribution très spécifique

**Cas particulier autorisé:**
• 6 cartes d'atout (dont Valet + 9) obligatoires
• + 2 cartes de couleurs différentes
• Pour avoir exactement 3 couleurs à la main

**Exemples de mains conformes:**
• Valet♠ 9♠ As♠ 10♠ Roi♠ Dame♠ + As♥ + 10♦ (3 couleurs: ♠♥♦)
• Valet♣ 9♣ As♣ 10♣ 8♣ 7♣ + Roi♠ + Dame♥ (3 couleurs: ♣♠♥)

**💡 Conseil Sofiene:**
Respectez STRICTEMENT la limite de 3 couleurs! Comptez bien avant d'annoncer.""",
                
                130: """**📢 Recommandation officielle pour 130 points**

**CRITÈRE TRÈS STRICT:** Maximum 2 couleurs + Atouts Complets

**Configuration exceptionnellement requise:**
• Seulement 2 couleurs dans votre main
• Plus atouts complets obligatoires
• Configuration très rare et risquée

**Cas particulier autorisé:**
• 6 cartes d'atout (dont Valet + 9) obligatoires
• + 2 cartes de même couleur ≠ atout
• Pour avoir exactement 2 couleurs à la main

**Exemples de mains conformes:**
• Valet♠ 9♠ As♠ 10♠ Roi♠ Dame♠ + As♥ + 10♥ (2 couleurs: ♠♥)
• Valet♦ 9♦ As♦ 10♦ 8♦ 7♦ + Roi♣ + Dame♣ (2 couleurs: ♦♣)

**💡 Conseil Sofiene:**
Configuration très restrictive! Soyez absolument certain avant d'annoncer.""",
                
                140: """**📢 Recommandation officielle pour 140 points**

**CRITÈRE EXTRÊME:** L'adversaire ne peut avoir qu'un seul pli maximum

**Configuration exceptionnelle requise:**
• Main quasi-parfaite obligatoire
• Domination totale du jeu
• Quasi-certitude de remporter 7 plis sur 8 minimum
• Contrôle absolu de plusieurs couleurs

**Conditions d'annonce:**
• Main extraordinaire uniquement
• Expérience de jeu confirmée
• Évaluation très prudente nécessaire

**⚠️ TRÈS RISQUÉ - RÉSERVÉ AUX EXPERTS**

**💡 Conseil Sofiene:**
Annonce exceptionnelle pour mains parfaites. N'annoncez que si vous êtes certain à 95%!"""
            },
            'en': {
                90: """**📢 Official recommendation for 90 points**

**Mandatory criterion:** Minimum 2 Aces

**Detailed configuration:**
• Relatively weak but playable hand
• At least 2 Aces in your game (any suit)
• Defensive strategy acceptable
• Moderate risk

**Compliant hand examples:**
• Ace♠ Ace♥ + 6 other various cards
• Ace♦ Ace♣ + medium cards
• Ace♠ Ace♦ + some face cards

**💡 Sofiene's advice:**
Safe and recommended announcement for beginners. Ideal when unsure about your hand.""",
                
                100: """**📢 Official recommendation for 100 points**

**Official criterion:** "Generally as you wish"

**Detailed configuration:**
• Maximum flexibility in composition
• Balanced hand recommended
• Some trumps appreciated but not mandatory
• Total freedom of choice

**Compliant hand examples:**
• Free composition with good balance
• Mix of trumps and strong cards
• Hand without strict criteria

**💡 Sofiene's advice:**
Flexible announcement perfect for adapting to the game. Use your experience to judge.""",
                
                110: """**📢 Official recommendation for 110 points**

**MANDATORY CRITERION:** Complete Trumps

**Strictly required configuration:**
• Must be sure to collect all trump cards from start
• **Option 1:** (Jack, 9, Ace, 10) of trump minimum
• **Option 2:** (Jack, 9, Ace + 2+ other trump cards)
• Total confidence in trump control

**Compliant hand examples:**
• Jack♠ 9♠ Ace♠ 10♠ + 4 other cards
• Jack♥ 9♥ Ace♥ King♥ Queen♥ + 3 others
• Jack♦ 9♦ Ace♦ 10♦ 8♦ 7♦ + 2 others

**⚠️ WARNING:** Without complete trumps, almost certain failure!

**💡 Sofiene's advice:**
Only take this risk if absolutely certain of controlling all trumps.""",
                
                120: """**📢 Official recommendation for 120 points**

**MANDATORY CRITERION:** Maximum 3 colors + Complete Trumps

**Strictly required configuration:**
• Only 3 colors in your hand (among: hearts, clubs, diamonds, spades)
• Plus complete trumps of one of these colors
• Very specific distribution

**Authorized special case:**
• 6 trump cards (including Jack + 9) mandatory
• + 2 cards of different colors
• To have exactly 3 colors in hand

**Compliant hand examples:**
• Jack♠ 9♠ Ace♠ 10♠ King♠ Queen♠ + Ace♥ + 10♦ (3 colors: ♠♥♦)
• Jack♣ 9♣ Ace♣ 10♣ 8♣ 7♣ + King♠ + Queen♥ (3 colors: ♣♠♥)

**💡 Sofiene's advice:**
STRICTLY respect the 3-color limit! Count carefully before announcing.""",
                
                130: """**📢 Official recommendation for 130 points**

**VERY STRICT CRITERION:** Maximum 2 colors + Complete Trumps

**Exceptionally required configuration:**
• Only 2 colors in your hand
• Plus complete trumps mandatory
• Very rare and risky configuration

**Authorized special case:**
• 6 trump cards (including Jack + 9) mandatory
• + 2 cards of same color ≠ trump
• To have exactly 2 colors in hand

**Compliant hand examples:**
• Jack♠ 9♠ Ace♠ 10♠ King♠ Queen♠ + Ace♥ + 10♥ (2 colors: ♠♥)
• Jack♦ 9♦ Ace♦ 10♦ 8♦ 7♦ + King♣ + Queen♣ (2 colors: ♦♣)

**💡 Sofiene's advice:**
Very restrictive configuration! Be absolutely certain before announcing.""",
                
                140: """**📢 Official recommendation for 140 points**

**EXTREME CRITERION:** Opponent can have maximum one trick

**Exceptional configuration required:**
• Near-perfect hand mandatory
• Total game domination
• Near-certainty of winning 7 out of 8 tricks minimum
• Absolute control of several suits

**Announcement conditions:**
• Extraordinary hand only
• Confirmed game experience
• Very careful evaluation necessary

**⚠️ VERY RISKY - RESERVED FOR EXPERTS**

**💡 Sofiene's advice:**
Exceptional announcement for perfect hands. Only announce if 95% certain!"""
            }
        }
        
        return recommendations.get(language, recommendations['fr']).get(points, 
            f"Aucune recommandation pour {points} points." if language == 'fr' 
            else f"No recommendation for {points} points.")
    
    def get_announcement_conditions_enhanced(self, points: int, language: str = 'fr') -> str:
        """Conditions d'annonces améliorées"""
        conditions = {
            'fr': {
                90: """**🎯 Quand annoncer 90 points:**

**Conditions idéales:**
• Vous avez au moins 2 As (obligatoire)
• Main faible mais pas catastrophique
• Stratégie défensive envisageable
• Début de partie prudent

**Situations favorables:**
• Jeu équilibré sans dominante claire
• Partenaire potentiellement fort
• Adversaires semblent hésitants

**À éviter:**
• Main très faible sans As
• Aucun atout dans la couleur choisie
• Adversaires très confiants""",
                
                100: """**🎯 Quand annoncer 100 points:**

**Conditions idéales:**
• "Généralement comme tu veux" (règle officielle)
• Main équilibrée sans critère strict
• Flexibilité maximale souhaitée
• Bon feeling général

**Situations favorables:**
• Jeu moyen avec potentiel
• Incertitude sur la meilleure stratégie
• Adaptation nécessaire selon le déroulement

**À éviter:**
• Main exceptionnelle (visez plus haut)
• Main très faible (restez à 90)""",
                
                110: """**🎯 Quand annoncer 110 points:**

**Conditions OBLIGATOIRES:**
• Atouts complets absolument certains
• (Valet, 9, As, 10) minimum en main
• Confiance totale de collecter tous les atouts

**Situations favorables:**
• Vous dominez la couleur d'atout
• Main solide avec contrôle
• Partenaire peut vous soutenir

**À éviter absolument:**
• Doute sur vos atouts
• Atouts incomplets
• Adversaires semblent forts dans votre couleur""",
                
                120: """**🎯 Quand annoncer 120 points:**

**Conditions STRICTES:**
• Maximum 3 couleurs à la main (compter!)
• Atouts complets obligatoires
• Configuration très spécifique requise

**Situations favorables:**
• Main concentrée sur 3 couleurs max
• Domination claire de l'atout
• Distribution exceptionnelle

**À éviter absolument:**
• 4 couleurs dans votre main
• Atouts incomplets
• Doute sur le comptage des couleurs""",
                
                130: """**🎯 Quand annoncer 130 points:**

**Conditions TRÈS STRICTES:**
• Maximum 2 couleurs à la main seulement
• Atouts complets obligatoires
• Configuration exceptionnellement rare

**Situations favorables:**
• Main bicolore avec domination
• Contrôle total de l'atout
• Quasi-certitude de réussite

**À éviter absolument:**
• Plus de 2 couleurs
• Atouts incomplets
• Moindre incertitude""",
                
                140: """**🎯 Quand annoncer 140 points:**

**Conditions EXTRÊMES:**
• Main quasi-parfaite uniquement
• Adversaire max 1 pli possible
• Domination totale évidente

**Situations favorables:**
• Main exceptionnelle rare
• Contrôle absolu du jeu
• Expérience confirmée

**À éviter absolument:**
• Moindre doute
• Main "juste" très bonne
• Manque d'expérience"""
            },
            'en': {
                90: """**🎯 When to announce 90 points:**

**Ideal conditions:**
• You have at least 2 Aces (mandatory)
• Weak but not catastrophic hand
• Defensive strategy feasible
• Cautious game start

**Favorable situations:**
• Balanced game without clear dominance
• Potentially strong partner
• Opponents seem hesitant

**To avoid:**
• Very weak hand without Aces
• No trumps in chosen suit
• Very confident opponents""",
                
                100: """**🎯 When to announce 100 points:**

**Ideal conditions:**
• "Generally as you wish" (official rule)
• Balanced hand without strict criteria
• Maximum flexibility desired
• Good general feeling

**Favorable situations:**
• Average game with potential
• Uncertainty about best strategy
• Adaptation needed according to progress

**To avoid:**
• Exceptional hand (aim higher)
• Very weak hand (stay at 90)""",
                
                110: """**🎯 When to announce 110 points:**

**MANDATORY conditions:**
• Complete trumps absolutely certain
• (Jack, 9, Ace, 10) minimum in hand
• Total confidence to collect all trumps

**Favorable situations:**
• You dominate the trump suit
• Solid hand with control
• Partner can support you

**Absolutely avoid:**
• Doubt about your trumps
• Incomplete trumps
• Opponents seem strong in your suit""",
                
                120: """**🎯 When to announce 120 points:**

**STRICT conditions:**
• Maximum 3 colors in hand (count!)
• Complete trumps mandatory
• Very specific configuration required

**Favorable situations:**
• Hand concentrated on 3 colors max
• Clear trump domination
• Exceptional distribution

**Absolutely avoid:**
• 4 colors in your hand
• Incomplete trumps
• Doubt about color counting""",
                
                130: """**🎯 When to announce 130 points:**

**VERY STRICT conditions:**
• Maximum 2 colors in hand only
• Complete trumps mandatory
• Exceptionally rare configuration

**Favorable situations:**
• Bicolor hand with domination
• Total trump control
• Near-certainty of success

**Absolutely avoid:**
• More than 2 colors
• Incomplete trumps
• Slightest uncertainty""",
                
                140: """**🎯 When to announce 140 points:**

**EXTREME conditions:**
• Near-perfect hand only
• Opponent max 1 possible trick
• Total obvious domination

**Favorable situations:**
• Rare exceptional hand
• Absolute game control
• Confirmed experience

**Absolutely avoid:**
• Slightest doubt
• "Just" very good hand
• Lack of experience"""
            }
        }
        
        return conditions.get(language, conditions['fr']).get(points, 
            f"Conditions pour {points} points non définies." if language == 'fr' 
            else f"Conditions for {points} points not defined.")
    
    def get_belote_detailed_info(self, language: str = 'fr') -> str:
        """Informations détaillées Belote/Rebelote"""
        rule = self.rules_db.get_all_rules()['belote_rebelote_detailed']
        title = rule['title_fr'] if language == 'fr' else rule['title_en']
        content = rule['content_fr'] if language == 'fr' else rule['content_en']
        return f"**{title}**\n\n{content}"
    
    def get_coinche_detailed_info(self, language: str = 'fr') -> str:
        """Informations détaillées Coinche/Surcoinche"""
        rule = self.rules_db.get_all_rules()['coinche_system_detailed']
        title = rule['title_fr'] if language == 'fr' else rule['title_en']
        content = rule['content_fr'] if language == 'fr' else rule['content_en']
        return f"**{title}**\n\n{content}"
    
    def get_capot_detailed_info(self, language: str = 'fr') -> str:
        """Informations détaillées Capot"""
        rule = self.rules_db.get_all_rules()['capot_rules_complete']
        title = rule['title_fr'] if language == 'fr' else rule['title_en']
        content = rule['content_fr'] if language == 'fr' else rule['content_en']
        return f"**{title}**\n\n{content}"
    
    def generate_enhanced_response(self, matches: List[RuleMatch], query: str, language: str = 'fr') -> str:
        """Générer une réponse améliorée"""
        if not matches:
            return self.intelligent_fallback(query, language)
        
        best_match = matches[0]
        rule = best_match.rule_data
        
        title = rule['title_fr'] if language == 'fr' else rule['title_en']
        content = rule['content_fr'] if language == 'fr' else rule['content_en']
        
        response = f"**{title}**\n\n{content}"
        
        # Ajouter informations sur la qualité de la correspondance
        if best_match.score > 0.9:
            confidence = "Excellente" if language == 'fr' else "Excellent"
        elif best_match.score > 0.7:
            confidence = "Bonne" if language == 'fr' else "Good"
        else:
            confidence = "Correcte" if language == 'fr' else "Correct"
        
        # Ajouter conseils d'expert pour certaines catégories
        if rule['category'] == 'announcements' and best_match.score > 0.8:
            expert_tip = f"\n\n**💡 Conseil d'expert Sofiene:**\n• Respectez strictement les critères officiels\n• En cas de doute, optez pour une annonce plus conservatrice\n• Observez le jeu des adversaires pour ajuster votre stratégie" if language == 'fr' else f"\n\n**💡 Sofiene expert tip:**\n• Strictly follow official criteria\n• When in doubt, choose more conservative announcement\n• Observe opponents' game to adjust your strategy"
            response += expert_tip
        
        # Ajouter suggestions de règles connexes
        if len(matches) > 1 and best_match.score > 0.8:
            related_header = "**📚 Voir aussi:**" if language == 'fr' else "**📚 See also:**"
            response += f"\n\n{related_header}\n"
            for match in matches[1:3]:
                related_title = match.rule_data['title_fr'] if language == 'fr' else match.rule_data['title_en']
                response += f"• {related_title}\n"
        
        return response
    
    def _cache_response(self, cache_key: str, response: str):
        """Mettre en cache une réponse"""
        if len(self.query_cache) >= self.max_cache_size:
            # Supprimer les entrées les plus anciennes
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        self.query_cache[cache_key] = response

class EnhancedConversationManager:
    """Gestionnaire de conversation amélioré"""
    
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.context_window = 7
        self.conversation_stats = {
            'total_queries': 0,
            'successful_responses': 0,
            'categories_discussed': set(),
            'start_time': datetime.now()
        }
        
    def add_message(self, sender: str, content: str, metadata: Dict = None):
        """Ajouter un message avec métadonnées"""
        message = {
            'sender': sender,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.messages.append(message)
        
        # Mettre à jour les statistiques
        if sender == 'user':
            self.conversation_stats['total_queries'] += 1
        elif sender == 'bot':
            self.conversation_stats['successful_responses'] += 1
        
        # Limiter la taille de l'historique
        if len(self.messages) > self.context_window * 2:
            self.messages = self.messages[-self.context_window * 2:]
    
    def get_enhanced_context(self) -> Dict[str, Any]:
        """Obtenir un contexte enrichi"""
        recent_messages = self.messages[-self.context_window:]
        user_messages = [msg['content'] for msg in recent_messages if msg['sender'] == 'user']
        
        # Analyser les sujets abordés
        topics = set()
        for msg in recent_messages:
            if 'category' in msg.get('metadata', {}):
                topics.add(msg['metadata']['category'])
        
        return {
            'recent_queries': user_messages,
            'discussed_topics': list(topics),
            'conversation_length': len(self.messages),
            'session_stats': self.conversation_stats
        }
    
    def export_enhanced_conversation(self, filename: str, language: str = 'fr'):
        """Export amélioré de la conversation"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # En-tête amélioré
                header = "=== Conversation Sofiene Expert Belote Contrée ===" if language == 'fr' else "=== Sofiene Belote Contrée Expert Conversation ==="
                f.write(f"{header}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Durée: {datetime.now() - self.conversation_stats['start_time']}\n")
                f.write(f"Requêtes totales: {self.conversation_stats['total_queries']}\n")
                f.write(f"Réponses fournies: {self.conversation_stats['successful_responses']}\n")
                f.write(f"Sujets abordés: {', '.join(self.conversation_stats['categories_discussed'])}\n\n")
                
                # Messages avec métadonnées
                for msg in self.messages:
                    timestamp = datetime.fromisoformat(msg['timestamp']).strftime('%H:%M:%S')
                    sender_label = "Vous" if msg['sender'] == 'user' and language == 'fr' else \
                                  "You" if msg['sender'] == 'user' else \
                                  "Sofiene Expert"
                    
                    f.write(f"[{timestamp}] {sender_label}:\n{msg['content']}\n\n")
                
                # Statistiques finales
                success_rate = (self.conversation_stats['successful_responses'] / max(1, self.conversation_stats['total_queries'])) * 100
                f.write(f"\n--- Statistiques de session ---\n")
                f.write(f"Taux de réussite: {success_rate:.1f}%" if language == 'fr' else f"Success rate: {success_rate:.1f}%")
                
            return True
        except Exception as e:
            st.error(f"Erreur d'export: {str(e)}")
            return False

def init_enhanced_session_state():
    """Initialiser l'état de session amélioré"""
    if 'conversation' not in st.session_state:
        st.session_state.conversation = EnhancedConversationManager()
    if 'ai' not in st.session_state:
        st.session_state.ai = EnhancedSofieneAI()
        st.session_state.ai.rule_embeddings = st.session_state.ai.initialize_embeddings()
    if 'language' not in st.session_state:
        st.session_state.language = 'fr'
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            'expert_mode': False,
            'show_confidence': True,
            'detailed_analysis': True
        }

def get_enhanced_suggestions(language: str):
    """Suggestions améliorées avec catégorisation"""
    if language == 'fr':
        return {
            "🎯 Annonces": [
                "Recommandation pour 120 points avec exemples",
                "Quand annoncer 110 points exactement?",
                "Différence entre 120 et 130 points",
                "Règles strictes pour 140 points"
            ],
            "👑 Belote/Rebelote": [
                "Stratégies avancées belote rebelote",
                "Quand utiliser roi dame atout?",
                "Timing optimal pour belote rebelote"
            ],
            "🏆 Techniques": [
                "Règles complètes du capot",
                "Système coinche surcoinche détaillé",
                "Calcul avancé des scores"
            ],
            "🔍 Évaluation": [
                "J'ai Valet, 9, As et 10 carreau plus 4 autres cartes",
                "Analyser ma main: As cœur, As trèfle, Roi pique",
                "Évaluation experte de main complexe"
            ]
        }
    else:
        return {
            "🎯 Announcements": [
                "Recommendation for 120 points with examples",
                "When to announce 110 points exactly?",
                "Difference between 120 and 130 points",
                "Strict rules for 140 points"
            ],
            "👑 Belote/Rebelote": [
                "Advanced belote rebelote strategies",
                "When to use king queen trump?",
                "Optimal timing for belote rebelote"
            ],
            "🏆 Techniques": [
                "Complete capot rules",
                "Detailed coinche surcoinche system",
                "Advanced score calculation"
            ],
            "🔍 Evaluation": [
                "I have Jack, 9, Ace and 10 diamonds plus 4 other cards",
                "Analyze my hand: Ace hearts, Ace clubs, King spades",
                "Expert evaluation of complex hand"
            ]
        }

def process_enhanced_message(message: str):
    """Traiter un message avec l'IA améliorée"""
    st.session_state.messages.append({"role": "user", "content": message})
    
    # Ajouter métadonnées
    metadata = {
        'query_length': len(message),
        'timestamp': datetime.now().isoformat()
    }
    st.session_state.conversation.add_message("user", message, metadata)
    
    # Obtenir le contexte enrichi
    context = st.session_state.conversation.get_enhanced_context()
    
    # Traiter avec l'IA améliorée
    response = st.session_state.ai.process_query_enhanced(
        message, 
        st.session_state.language, 
        context['recent_queries']
    )
    
    # Déterminer la catégorie pour les métadonnées
    intent = st.session_state.ai.extract_intent_enhanced(message, st.session_state.language)
    response_metadata = {
        'category': intent,
        'confidence': 'high',  # Pourrait être calculé
        'response_length': len(response)
    }
    
    st.session_state.conversation.add_message("bot", response, response_metadata)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Mettre à jour les statistiques
    st.session_state.conversation.conversation_stats['categories_discussed'].add(intent)

def main_enhanced():
    """Application Streamlit principale améliorée"""
    
    st.set_page_config(
        page_title="Sofiene Expert - Belote Tunisienne Contrée",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_enhanced_session_state()
    
    # CSS amélioré
    st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        margin-bottom: 5px;
        border-radius: 20px;
        border: 2px solid #1f4e79;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #1f4e79;
        color: white;
        transform: translateY(-2px);
    }
    .sofiene-header {
        background: linear-gradient(135deg, #1f4e79, #2d5aa0, #3a6bb3);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .expert-badge {
        background: linear-gradient(45deg, #28a745, #20c997);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
        box-shadow: 0 2px 10px rgba(40,167,69,0.3);
    }
    .suggestion-category {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        border-left: 4px solid #1f4e79;
    }
    .stats-card {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .footer-dev {
        background: linear-gradient(135deg, #f0f2f6, #e1e5e9);
        padding: 1rem;
        border-radius: 10px;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 1rem;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar améliorée
    with st.sidebar:
        st.markdown("""
        <div class="sofiene-header">
            <h1>🎮 Sofiene Expert</h1>
            <p>Expert en Belote Tunisienne Contrée</p>
            <span class="expert-badge">IA Avancée</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Sélecteur de langue amélioré
        current_lang = st.session_state.language
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"🇫🇷 {'FR' if current_lang == 'en' else 'Français'}", key="lang_fr"):
                st.session_state.language = 'fr'
                st.rerun()
        with col2:
            if st.button(f"🇬🇧 {'EN' if current_lang == 'fr' else 'English'}", key="lang_en"):
                st.session_state.language = 'en'
                st.rerun()
        
        st.divider()
        
        # Préférences utilisateur
        if st.session_state.language == 'fr':
            st.subheader("⚙️ Préférences")
            st.session_state.user_preferences['expert_mode'] = st.checkbox("Mode expert", value=st.session_state.user_preferences['expert_mode'])
            st.session_state.user_preferences['detailed_analysis'] = st.checkbox("Analyses détaillées", value=st.session_state.user_preferences['detailed_analysis'])
        else:
            st.subheader("⚙️ Preferences")
            st.session_state.user_preferences['expert_mode'] = st.checkbox("Expert mode", value=st.session_state.user_preferences['expert_mode'])
            st.session_state.user_preferences['detailed_analysis'] = st.checkbox("Detailed analysis", value=st.session_state.user_preferences['detailed_analysis'])
        
        st.divider()
        
        # Suggestions catégorisées
        suggestions_title = "💡 Suggestions par catégorie:" if st.session_state.language == 'fr' else "💡 Suggestions by category:"
        st.subheader(suggestions_title)
        
        suggestions = get_enhanced_suggestions(st.session_state.language)
        for category, items in suggestions.items():
            with st.expander(category):
                for i, suggestion in enumerate(items):
                    if st.button(suggestion, key=f"sug_{category}_{i}_{st.session_state.language}"):
                        process_enhanced_message(suggestion)
                        st.rerun()
        
        st.divider()
        
        # Statistiques de session
        if st.session_state.language == 'fr':
            st.subheader("📊 Statistiques de session")
        else:
            st.subheader("📊 Session statistics")
        
        stats = st.session_state.conversation.conversation_stats
        
        st.markdown(f"""
        <div class="stats-card">
            <strong>Questions posées:</strong> {stats['total_queries']}<br>
            <strong>Réponses fournies:</strong> {stats['successful_responses']}<br>
            <strong>Sujets abordés:</strong> {len(stats['categories_discussed'])}
        </div>
        """, unsafe_allow_html=True)
        
        # Export amélioré
        if st.button("💾 " + ("Exporter conversation" if st.session_state.language == 'fr' else "Export conversation")):
            filename = f"sofiene_expert_conversation_{st.session_state.conversation.conversation_stats['start_time'].strftime('%Y%m%d_%H%M%S')}.txt"
            if st.session_state.conversation.export_enhanced_conversation(filename, st.session_state.language):
                st.success(f"✅ Exporté: {filename}" if st.session_state.language == 'fr' else f"✅ Exported: {filename}")
                
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        st.download_button(
                            label="📥 " + ("Télécharger" if st.session_state.language == 'fr' else "Download"),
                            data=f.read(),
                            file_name=filename,
                            mime="text/plain"
                        )
                except Exception as e:
                    st.warning(f"⚠️ Erreur de téléchargement: {str(e)}" if st.session_state.language == 'fr' 
                              else f"⚠️ Download error: {str(e)}")
        
        # Footer développeur amélioré
        st.markdown("""
        <div class="footer-dev">
            <p><strong>🚀 Sofiene Expert v2.0</strong></p>
            <p>Développé par <strong>BellaajMohsen7</strong></p>
            <p>IA Avancée • Compréhension Linguistique • Expertise Complète</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Contenu principal amélioré
    if st.session_state.language == 'fr':
        st.title("🎮 Sofiene Expert - Belote Tunisienne Contrée")
        st.markdown("""
        **🧠 Assistant IA avancé pour maîtriser la Belote Contrée**
        
        Sofiene Expert utilise une intelligence artificielle avancée avec compréhension linguistique améliorée 
        pour vous accompagner dans tous les aspects de la Belote Tunisienne Contrée.
        
        **🎯 Nouvelles capacités:**
        • Compréhension de variations linguistiques ("règle d'annonce", "que annoncer", etc.)
        • Analyse experte de main avec recommandations détaillées
        • Base de données complète des règles officielles
        • Réponses contextuelles et adaptatives
        • Gestion des fautes de frappe et langage informel
        
        **🔥 Expertise disponible:**
        • Recommandations officielles pour tous les niveaux d'annonce (90-140)
        • Évaluation experte de vos mains avec analyse détaillée
        • Règles complètes Belote/Rebelote avec stratégies
        • Système de scoring officiel avec cas spéciaux
        • Coinche/Surcoinche et gestion des risques
        • Règles du Capot et situations exceptionnelles
        """)
    else:
        st.title("🎮 Sofiene Expert - Tunisian Belote Contrée")
        st.markdown("""
        **🧠 Advanced AI assistant to master Belote Contrée**
        
        Sofiene Expert uses advanced artificial intelligence with enhanced linguistic understanding 
        to accompany you in all aspects of Tunisian Belote Contrée.
        
        **🎯 New capabilities:**
        • Understanding of linguistic variations ("announcement rule", "what to announce", etc.)
        • Expert hand analysis with detailed recommendations
        • Complete database of official rules
        • Contextual and adaptive responses
        • Handling of typos and informal language
        
        **🔥 Available expertise:**
        • Official recommendations for all announcement levels (90-140)
        • Expert evaluation of your hands with detailed analysis
        • Complete Belote/Rebelote rules with strategies
        • Official scoring system with special cases
        • Coinche/Surcoinche and risk management
        • Capot rules and exceptional situations
        """)
    
    # Section de démonstration améliorée
    if st.session_state.language == 'fr':
        with st.expander("🚀 Testez les nouvelles capacités de Sofiene Expert"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🎯 Compréhension linguistique:**
                • "règle d'annonce" ou "regle annonce"
                • "que annoncer avec ma main?"
                • "calculer point" ou "calcul score"
                • "quand utiliser belote rebelote"
                """)
            
            with col2:
                st.markdown("""
                **🔍 Évaluation avancée:**
                • "J'ai Valet, 9, As carreau, que annoncer?"
                • "Main avec 6 atouts dont Valet et 9"
                • "As cœur, As trèfle, Roi pique, conseil?"
                • "Analyser ma main complexe"
                """)
    else:
        with st.expander("🚀 Test Sofiene Expert's new capabilities"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🎯 Linguistic understanding:**
                • "announcement rule" or "announce rule"
                • "what to announce with my hand?"
                • "calculate point" or "score calculation"
                • "when to use belote rebelote"
                """)
            
            with col2:
                st.markdown("""
                **🔍 Advanced evaluation:**
                • "I have Jack, 9, Ace diamonds, what to announce?"
                • "Hand with 6 trumps including Jack and 9"
                • "Ace hearts, Ace clubs, King spades, advice?"
                • "Analyze my complex hand"
                """)
    
    # Interface de chat
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Zone de saisie améliorée
    if st.session_state.language == 'fr':
        prompt_text = "Posez votre question sur la Belote Contrée... (Sofiene comprend maintenant les variations!)"
    else:
        prompt_text = "Ask your Belote Contrée question... (Sofiene now understands variations!)"
    
    if prompt := st.chat_input(prompt_text):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🧠 Sofiene Expert analyse..." if st.session_state.language == 'fr' else "🧠 Sofiene Expert analyzing..."):
                try:
                    process_enhanced_message(prompt)
                    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                        st.markdown(st.session_state.messages[-1]["content"])
                        st.rerun()
                        
                except Exception as e:
                    error_msg = f"🚨 Erreur d'analyse: {str(e)}" if st.session_state.language == 'fr' else f"🚨 Analysis error: {str(e)}"
                    st.error(error_msg)
                    
                    # Message de fallback amélioré
                    fallback_msg = """🔧 Je rencontre une difficulté technique temporaire. 

**Essayez:**
• Reformuler votre question différemment
• Utiliser des termes plus simples
• Poser une question plus spécifique

**Exemples qui fonctionnent:**
• "Recommandation pour 120 points"
• "Règles belote rebelote"
• "Calculer les scores"

Je suis là pour vous aider!""" if st.session_state.language == 'fr' else """🔧 I'm experiencing a temporary technical difficulty.

**Try:**
• Rephrase your question differently
• Use simpler terms
• Ask a more specific question

**Examples that work:**
• "Recommendation for 120 points"
• "Belote rebelote rules"
• "Calculate scores"

I'm here to help!"""
                    st.markdown(fallback_msg)
    
    # Footer principal amélioré
    st.divider()
    
    if st.session_state.language == 'fr':
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            **🎯 IA Avancée**
            • Compréhension linguistique
            • Gestion des variations
            • Apprentissage contextuel
            """)
        
        with col2:
            st.markdown("""
            **📚 Base Complète**
            • Toutes les règles officielles
            • Cas spéciaux et exceptions
            • Exemples pratiques
            """)
        
        with col3:
            st.markdown("""
            **🔍 Analyse Experte**
            • Évaluation de main détaillée
            • Recommandations précises
            • Stratégies optimales
            """)
        
        with col4:
            st.markdown("""
            **💡 Assistant Intelligent**
            • Réponses adaptatives
            • Suggestions contextuelles
            • Support multilingue
            """)
        
        st.markdown("""
        ---
        **🚀 Sofiene Expert v2.0 - Développé avec passion par BellaajMohsen7**  
        *Intelligence Artificielle Avancée pour la Belote Tunisienne Contrée*
        
        📧 Contact: BellaajMohsen7@github.com | 🌟 Version 2.0 Production | 🧠 IA Enhanced
        """)
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            **🎯 Advanced AI**
            • Linguistic understanding
            • Variation handling
            • Contextual learning
            """)
        
        with col2:
            st.markdown("""
            **📚 Complete Base**
            • All official rules
            • Special cases and exceptions
            • Practical examples
            """)
        
        with col3:
            st.markdown("""
            **🔍 Expert Analysis**
            • Detailed hand evaluation
            • Precise recommendations
            • Optimal strategies
            """)
        
        with col4:
            st.markdown("""
            **💡 Intelligent Assistant**
            • Adaptive responses
            • Contextual suggestions
            • Multilingual support
            """)
        
        st.markdown("""
        ---
        **🚀 Sofiene Expert v2.0 - Developed with passion by BellaajMohsen7**  
        *Advanced Artificial Intelligence for Tunisian Belote Contrée*
        
        📧 Contact: BellaajMohsen7@github.com | 🌟 Version 2.0 Production | 🧠 AI Enhanced
        """)

if __name__ == "__main__":
    main_enhanced()