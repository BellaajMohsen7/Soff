#!/usr/bin/env python3
"""
Sofiene - Expert en Belote Tunisienne Contrée
Bot intelligent basé sur les règles officielles
Développé par BellaajMohsen7 - 2025
"""

import streamlit as st
import numpy as np
import pickle
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Import required libraries with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    st.error("Veuillez installer les dépendances: pip install sentence-transformers scikit-learn")

@dataclass
class RuleMatch:
    rule_id: str
    score: float
    rule_data: Dict

@dataclass
class HandEvaluation:
    recommended_announcement: int
    confidence: float
    reasoning: str
    alternative_options: List[int]

class HandEvaluator:
    """Évaluateur de main expert pour les annonces Belote"""
    
    def __init__(self):
        self.trump_values = {
            'valet': 20, 'v': 20, 'j': 20,
            '9': 14,
            'as': 11, 'a': 11, 'ace': 11,
            '10': 10, 'dix': 10,
            'roi': 4, 'r': 4, 'k': 4, 'king': 4,
            'dame': 3, 'd': 3, 'q': 3, 'queen': 3,
            '8': 0, 'huit': 0,
            '7': 0, 'sept': 0
        }
        
        self.non_trump_values = {
            'as': 11, 'a': 11, 'ace': 11,
            '10': 10, 'dix': 10,
            'roi': 4, 'r': 4, 'k': 4, 'king': 4,
            'dame': 3, 'd': 3, 'q': 3, 'queen': 3,
            'valet': 2, 'v': 2, 'j': 2,
            '9': 0,
            '8': 0, 'huit': 0,
            '7': 0, 'sept': 0
        }
    
    def evaluate_hand_simple(self, description: str, language: str = 'fr') -> HandEvaluation:
        """Évaluation simple basée sur des patterns de description"""
        description_lower = description.lower()
        
        # Évaluation pour 110 points - Atouts complets détectés
        if self._matches_pattern(description_lower, [
            r'valet.*9.*as.*10', r'as.*10.*valet.*9', 
            r'valet.*9.*as.*\d+.*autres.*atout'
        ]):
            return HandEvaluation(
                recommended_announcement=110,
                confidence=0.9,
                reasoning="Atouts complets détectés - parfait pour 110 points selon les règles officielles",
                alternative_options=[100, 120]
            )
        
        # Évaluation pour 120 points - 6 atouts ou 3 couleurs max
        if self._matches_pattern(description_lower, [
            r'6.*cartes.*atout', r'6.*atouts', r'3.*couleurs'
        ]):
            return HandEvaluation(
                recommended_announcement=120,
                confidence=0.8,
                reasoning="Configuration pour 120 points: maximum 3 couleurs + atouts complets",
                alternative_options=[110, 130]
            )
        
        # Évaluation pour 90 points - 2 As minimum
        if self._matches_pattern(description_lower, [r'2.*as', r'deux.*as']):
            return HandEvaluation(
                recommended_announcement=90,
                confidence=0.8,
                reasoning="Avec 2 As minimum, 90 points est la recommandation officielle",
                alternative_options=[100]
            )
        
        # Évaluation conservative par défaut
        return HandEvaluation(
            recommended_announcement=90,
            confidence=0.6,
            reasoning="Recommandation conservatrice - analysez votre main selon les critères officiels",
            alternative_options=[100]
        )
    
    def _matches_pattern(self, text: str, patterns: List[str]) -> bool:
        """Vérifier si le texte correspond à un pattern"""
        return any(re.search(pattern, text) for pattern in patterns)

class BeloteRulesDatabase:
    """Base de données officielle des règles de Belote Contrée Tunisienne"""
    
    def __init__(self):
        self.rules = self._initialize_rules()
        
    def _initialize_rules(self):
        """Initialiser la base de données des règles officielles"""
        return {
            'announcements_official': {
                'id': 'announcements_official',
                'category': 'announcements',
                'title_fr': '📢 Règles Officielles des Annonces',
                'title_en': '📢 Official Announcement Rules',
                'content_fr': """**Règles officielles des annonces:**

**90 points:**
• **Recommandation:** 2 As minimum
• Configuration de base acceptable

**100 points:**
• **Recommandation:** "Généralement comme tu veux"
• Flexibilité dans la composition de la main

**110 points:**
• **Recommandation:** Atouts Complets OBLIGATOIRES
• Être sûr de collecter toutes les cartes d'atout dès le début
• **Exemples requis:**
  - (Valet, 9, As, 10) ou plus
  - (Valet, 9, As, 2+ autres cartes d'atout)

**120 points:**
• **Recommandation:** Seulement 3 couleurs à la main + Atouts Complets
• **Cas particulier:** 6 cartes d'atout (dont Valet + 9) + 2 autres cartes de couleurs différentes pour avoir 3 couleurs à la main

**130 points:**
• **Recommandation:** Seulement 2 couleurs à la main + Atouts Complets
• **Cas particulier:** 6 cartes d'atout (dont Valet + 9) + 2 cartes même couleur ≠ atout pour avoir 2 couleurs à la main

**140 points:**
• **Recommandation:** L'adversaire ne peut avoir qu'un seul pli au maximum
• Main quasi-parfaite requise""",
                'content_en': """**Official announcement rules:**

**90 points:**
• **Recommendation:** Minimum 2 Aces
• Basic acceptable configuration

**100 points:**
• **Recommendation:** "Generally as you wish"
• Flexibility in hand composition

**110 points:**
• **Recommendation:** Complete Trumps MANDATORY
• Must be sure to collect all trump cards from the start
• **Required examples:**
  - (Jack, 9, Ace, 10) or more
  - (Jack, 9, Ace, 2+ other trump cards)

**120 points:**
• **Recommendation:** Only 3 colors in hand + Complete Trumps
• **Special case:** 6 trump cards (including Jack + 9) + 2 other cards of different colors to have 3 colors in hand

**130 points:**
• **Recommendation:** Only 2 colors in hand + Complete Trumps
• **Special case:** 6 trump cards (including Jack + 9) + 2 cards same color ≠ trump to have 2 colors in hand

**140 points:**
• **Recommendation:** Opponent can have maximum one trick
• Near-perfect hand required""",
                'keywords_fr': ['annonce', 'recommandation', '90', '100', '110', '120', '130', '140', 'atouts', 'complets', 'couleurs', 'officiel'],
                'keywords_en': ['announcement', 'recommendation', '90', '100', '110', '120', '130', '140', 'trumps', 'complete', 'colors', 'official'],
                'patterns_fr': [
                    r'recommandation.*?(?:pour|de).*?(\d{2,3})',
                    r'(\d{2,3}).*points.*recommandation',
                    r'quand.*annoncer.*?(\d{2,3})',
                    r'annoncer.*?(\d{2,3})',
                    r'contrat.*?(\d{2,3})'
                ],
                'patterns_en': [
                    r'recommendation.*?(?:for|of).*?(\d{2,3})',
                    r'(\d{2,3}).*points.*recommendation',
                    r'when.*announce.*?(\d{2,3})',
                    r'announce.*?(\d{2,3})',
                    r'contract.*?(\d{2,3})'
                ]
            },
            
            'scoring_official': {
                'id': 'scoring_official',
                'category': 'scoring',
                'title_fr': '🔢 Système de Score Officiel',
                'title_en': '🔢 Official Scoring System',
                'content_fr': """**Calcul officiel des scores:**

**Total des points possibles:**
• Points des cartes: 152
• Dix de der (dernier pli): +10 points
• **Total possible: 162 points**

**Système de score spécial pour équipe non-preneuse:**
Si score = 10×K + x:
• Si x ∈ [5,6,7] → Score final = 10×(K+1)
• Sinon → Score final = 10×K
• Autre équipe: 160 - score calculé

**Bonus Belote/Rebelote:**
• +20 points si Roi et Dame d'atout chez même joueur

**En cas d'échec de contrat:**
• Équipe preneuse: 0 points
• Équipe adverse: 160 + 20×(bonus belote)

**Capot:**
• Tous les plis = 250 points automatiques
• Si dans contrat: DOIT faire tous les plis

**Coinche & Surcoinche:**
• Contrat simple: ×1
• Coinché: ×2
• Surcoinché: ×4""",
                'content_en': """**Official score calculation:**

**Total possible points:**
• Card points: 152
• Ten of last (last trick): +10 points
• **Total possible: 162 points**

**Special scoring system for non-taking team:**
If score = 10×K + x:
• If x ∈ [5,6,7] → Final score = 10×(K+1)
• Otherwise → Final score = 10×K
• Other team: 160 - calculated score

**Belote/Rebelote bonus:**
• +20 points if King and Queen of trump with same player

**Contract failure:**
• Taking team: 0 points
• Opposing team: 160 + 20×(belote bonus)

**Capot:**
• All tricks = 250 automatic points
• If in contract: MUST make all tricks

**Coinche & Surcoinche:**
• Simple contract: ×1
• Coinched: ×2
• Surcoinched: ×4""",
                'keywords_fr': ['score', 'points', 'calcul', 'officiel', 'système', 'capot', 'coinche', 'belote', 'rebelote'],
                'keywords_en': ['score', 'points', 'calculation', 'official', 'system', 'capot', 'coinche', 'belote', 'rebelote'],
                'patterns_fr': [
                    r'calculer.*points',
                    r'score.*système',
                    r'comment.*compter'
                ],
                'patterns_en': [
                    r'calculate.*points',
                    r'scoring.*system',
                    r'how.*count'
                ]
            },
            
            'belote_rebelote_official': {
                'id': 'belote_rebelote_official',
                'category': 'bonus',
                'title_fr': '👑 Belote et Rebelote Officiel',
                'title_en': '👑 Official Belote and Rebelote',
                'content_fr': """**Règles officielles Belote/Rebelote:**

**Définition:**
• Avoir le Roi ET la Dame d'atout chez le même joueur
• Bonus: +20 points à l'équipe

**Quand l'utiliser:**
• Annoncez "Belote" en jouant la première carte (Roi ou Dame)
• Annoncez "Rebelote" en jouant la seconde carte
• L'annonce est OBLIGATOIRE pour obtenir les points
• Si oubli d'annoncer = pas de bonus

**Règles d'annonce:**
• Peut être joué à tout moment du jeu
• L'ordre Roi puis Dame n'est pas obligatoire
• Valable uniquement si les deux cartes sont chez même joueur

**Calcul dans le score:**
• +20 points ajoutés au score de l'équipe
• Compte dans le calcul final des contrats
• Peut faire la différence dans un contrat serré

**Stratégie:**
• Gardez ces cartes pour moments cruciaux
• Utilisez pour remporter plis importants
• Coordination avec partenaire essentielle""",
                'content_en': """**Official Belote/Rebelote rules:**

**Definition:**
• Having King AND Queen of trump with same player
• Bonus: +20 points to the team

**When to use:**
• Announce "Belote" when playing first card (King or Queen)
• Announce "Rebelote" when playing second card
• Announcement is MANDATORY to get points
• If forgotten = no bonus

**Announcement rules:**
• Can be played anytime during game
• King then Queen order not mandatory
• Valid only if both cards with same player

**Score calculation:**
• +20 points added to team score
• Counts in final contract calculation
• Can make difference in tight contract

**Strategy:**
• Keep these cards for crucial moments
• Use to win important tricks
• Partner coordination essential""",
                'keywords_fr': ['belote', 'rebelote', 'roi', 'dame', 'atout', 'bonus', '20', 'points', 'officiel', 'utiliser'],
                'keywords_en': ['belote', 'rebelote', 'king', 'queen', 'trump', 'bonus', '20', 'points', 'official', 'use'],
                'patterns_fr': [
                    r'belote.*rebelote',
                    r'quand.*utiliser.*belote',
                    r'comment.*belote',
                    r'roi.*dame.*atout'
                ],
                'patterns_en': [
                    r'belote.*rebelote',
                    r'when.*use.*belote',
                    r'how.*belote',
                    r'king.*queen.*trump'
                ]
            },
            
            'card_values_official': {
                'id': 'card_values_official',
                'category': 'cards',
                'title_fr': '🃏 Valeurs Officielles des Cartes',
                'title_en': '🃏 Official Card Values',
                'content_fr': """**Valeurs officielles:**

**À l'atout:**
• Valet: 20 points 🏆 (carte la plus forte)
• 9: 14 points
• As: 11 points
• 10: 10 points
• Roi: 4 points
• Dame: 3 points
• 8, 7: 0 point

**Hors atout:**
• As: 11 points 🏆 (carte la plus forte)
• 10: 10 points
• Roi: 4 points
• Dame: 3 points
• Valet: 2 points
• 9, 8, 7: 0 point

**Totaux officiels:**
• Points cartes: 152 maximum
• Dix de der: +10 points
• **Total possible: 162 points par manche**

**Ordre de force à l'atout:**
Valet > 9 > As > 10 > Roi > Dame > 8 > 7

**Ordre de force hors atout:**
As > 10 > Roi > Dame > Valet > 9 > 8 > 7""",
                'content_en': """**Official values:**

**Trump cards:**
• Jack: 20 points 🏆 (strongest card)
• 9: 14 points
• Ace: 11 points
• 10: 10 points
• King: 4 points
• Queen: 3 points
• 8, 7: 0 points

**Non-trump cards:**
• Ace: 11 points 🏆 (strongest card)
• 10: 10 points
• King: 4 points
• Queen: 3 points
• Jack: 2 points
• 9, 8, 7: 0 points

**Official totals:**
• Card points: 152 maximum
• Ten of last: +10 points
• **Total possible: 162 points per round**

**Trump strength order:**
Jack > 9 > Ace > 10 > King > Queen > 8 > 7

**Non-trump strength order:**
Ace > 10 > King > Queen > Jack > 9 > 8 > 7""",
                'keywords_fr': ['valeurs', 'cartes', 'atout', 'valet', 'as', 'points', 'officiel', 'ordre'],
                'keywords_en': ['values', 'cards', 'trump', 'jack', 'ace', 'points', 'official', 'order'],
                'patterns_fr': [
                    r'valeur.*carte',
                    r'combien.*points',
                    r'carte.*forte'
                ],
                'patterns_en': [
                    r'card.*value',
                    r'how.*points',
                    r'strongest.*card'
                ]
            },
            
            'basic_rules_official': {
                'id': 'basic_rules_official',
                'category': 'basic',
                'title_fr': '🎮 Règles Officielles de Base',
                'title_en': '🎮 Official Basic Rules',
                'content_fr': """**Configuration officielle:**
• 4 joueurs organisés en 2 équipes de 2
• Jeu de 32 cartes (7 à As)
• 8 cartes distribuées par joueur

**Annonce obligatoire:**
• Chaque joueur peut annoncer un contrat (ex: "120 Cœur")
• L'annonce la plus forte détermine l'atout
• Si personne n'annonce → redistribution des cartes

**Déroulement d'un tour:**
• Joueur à droite du donneur commence
• Obligation de suivre la couleur demandée
• Si impossible, peut jouer n'importe quelle carte
• Pli remporté par carte la plus forte ou atout le plus fort

**Fin de partie officielle:**
• Partie jouée en plusieurs manches
• Premier à atteindre 1001 points ou plus remporte
• Alternative: 2000 points selon accord

**Règles de distribution:**
• 8 cartes par joueur, distribuées en une fois
• Pas de cartes retournées
• Annonces dans le sens horaire""",
                'content_en': """**Official configuration:**
• 4 players organized in 2 teams of 2
• 32-card deck (7 to Ace)
• 8 cards dealt per player

**Mandatory announcement:**
• Each player can announce contract (ex: "120 Hearts")
• Highest announcement determines trump
• If no one announces → cards redistributed

**Turn progression:**
• Player to right of dealer starts
• Must follow requested suit
• If impossible, can play any card
• Trick won by highest card or strongest trump

**Official game end:**
• Game played over several rounds
• First to reach 1001 points or more wins
• Alternative: 2000 points by agreement

**Distribution rules:**
• 8 cards per player, dealt at once
• No cards turned over
• Announcements clockwise""",
                'keywords_fr': ['règles', 'base', 'officiel', 'jeu', 'configuration', 'distribution', 'tour'],
                'keywords_en': ['rules', 'basic', 'official', 'game', 'configuration', 'distribution', 'turn'],
                'patterns_fr': [
                    r'règles.*base',
                    r'comment.*jouer',
                    r'début.*jeu'
                ],
                'patterns_en': [
                    r'basic.*rules',
                    r'how.*play',
                    r'start.*game'
                ]
            }
        }
        
    def get_all_rules(self):
        """Retourner toutes les règles"""
        return self.rules

class ConversationManager:
    """Gestionnaire de conversation"""
    
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.context_window = 5
        
    def add_message(self, sender: str, content: str):
        """Ajouter un message à l'historique"""
        message = {
            'sender': sender,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        self.messages.append(message)
        
        if len(self.messages) > self.context_window * 2:
            self.messages = self.messages[-self.context_window * 2:]
            
    def get_context(self) -> List[str]:
        """Obtenir le contexte récent"""
        recent_messages = self.messages[-self.context_window:]
        return [msg['content'] for msg in recent_messages if msg['sender'] == 'user']
        
    def clear_history(self):
        """Effacer l'historique"""
        self.messages.clear()
        
    def get_timestamp(self) -> str:
        """Obtenir timestamp actuel"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def export_to_file(self, filename: str, language: str = 'fr'):
        """Exporter la conversation"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                header = "Conversation Sofiene Bot" if language == 'fr' else "Sofiene Bot Conversation"
                f.write(f"=== {header} ===\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for msg in self.messages:
                    sender_label = "Vous" if msg['sender'] == 'user' and language == 'fr' else \
                                  "You" if msg['sender'] == 'user' else \
                                  "Sofiene"
                    f.write(f"{sender_label}: {msg['content']}\n\n")
            return True
        except Exception as e:
            st.error(f"Erreur d'export: {str(e)}")
            return False
                
    def get_conversation_summary(self, language: str = 'fr') -> str:
        """Résumé de la conversation"""
        if not self.messages:
            return "Aucune conversation" if language == 'fr' else "No conversation"
            
        user_msg_count = len([m for m in self.messages if m['sender'] == 'user'])
        bot_msg_count = len([m for m in self.messages if m['sender'] == 'bot'])
        
        if language == 'fr':
            return f"Conversation: {user_msg_count} questions, {bot_msg_count} réponses"
        else:
            return f"Conversation: {user_msg_count} questions, {bot_msg_count} responses"

@st.cache_resource
def load_sentence_transformer():
    """Charger le modèle de transformation de phrases"""
    if DEPENDENCIES_AVAILABLE:
        try:
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Erreur de chargement du modèle: {str(e)}")
            return None
    return None

class SofieneAI:
    """Sofiene - Expert IA en Belote Tunisienne Contrée"""
    
    def __init__(self):
        self.model = load_sentence_transformer()
        self.rules_db = BeloteRulesDatabase()
        self.hand_evaluator = HandEvaluator()
        self.rule_embeddings = {}
        self.context_window = 3
        
        if self.model:
            self.initialize_embeddings()
        
    @st.cache_data
    def initialize_embeddings(_self):
        """Initialiser les embeddings"""
        embeddings_file = 'sofiene_embeddings.pkl'
        
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
            with st.spinner("Chargement de l'expertise Sofiene..."):
                for rule_id, rule in self.rules_db.get_all_rules().items():
                    text_fr = f"{rule['title_fr']} {rule['content_fr']} {' '.join(rule['keywords_fr'])}"
                    text_en = f"{rule['title_en']} {rule['content_en']} {' '.join(rule['keywords_en'])}"
                    
                    embedding_fr = self.model.encode(text_fr)
                    embedding_en = self.model.encode(text_en)
                    
                    embeddings[rule_id] = {
                        'fr': embedding_fr,
                        'en': embedding_en,
                        'rule': rule
                    }
                
                try:
                    with open('sofiene_embeddings.pkl', 'wb') as f:
                        pickle.dump(embeddings, f)
                except Exception:
                    pass
                    
            return embeddings
        except Exception as e:
            st.error(f"Erreur de traitement: {str(e)}")
            return {}
    
    def handle_specific_patterns(self, query: str, language: str = 'fr') -> Optional[str]:
        """Gérer les patterns spécifiques avec reconnaissance précise"""
        query_lower = query.lower().strip()
        
        # Patterns d'évaluation de main
        hand_patterns = {
            'fr': [
                r'j.ai.*(as|valet|roi|dame|10|9|8|7).*que.*annoncer',
                r'j.ai.*main.*annoncer',
                r'que.*annoncer.*avec.*(as|valet|roi|dame|10|9|8|7)',
                r'main.*(as|valet|roi|dame|10|9|8|7).*annoncer'
            ],
            'en': [
                r'i.have.*(ace|jack|king|queen|10|9|8|7).*what.*announce',
                r'i.have.*hand.*announce',
                r'what.*announce.*with.*(ace|jack|king|queen|10|9|8|7)',
                r'hand.*(ace|jack|king|queen|10|9|8|7).*announce'
            ]
        }
        
        # Vérifier les patterns d'évaluation de main
        patterns = hand_patterns.get(language, hand_patterns['fr'])
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return self.handle_hand_evaluation(query, language)
        
        # Patterns de recommandations d'annonces
        announcement_patterns = {
            'fr': [
                (r'recommandation.*?(?:pour|de).*?(\d{2,3})', self.get_announcement_recommendation),
                (r'(\d{2,3}).*points.*recommandation', self.get_announcement_recommendation),
                (r'quand.*annoncer.*?(\d{2,3})', self.get_announcement_conditions),
                (r'annoncer.*?(\d{2,3})', self.get_announcement_conditions),
                (r'contrat.*?(\d{2,3})', self.get_announcement_recommendation)
            ],
            'en': [
                (r'recommendation.*?(?:for|of).*?(\d{2,3})', self.get_announcement_recommendation),
                (r'(\d{2,3}).*points.*recommendation', self.get_announcement_recommendation),
                (r'when.*announce.*?(\d{2,3})', self.get_announcement_conditions),
                (r'announce.*?(\d{2,3})', self.get_announcement_conditions),
                (r'contract.*?(\d{2,3})', self.get_announcement_recommendation)
            ]
        }
        
        patterns = announcement_patterns.get(language, announcement_patterns['fr'])
        for pattern, handler in patterns:
            match = re.search(pattern, query_lower)
            if match:
                try:
                    points_str = match.group(1)
                    points = int(points_str)
                    if 90 <= points <= 140:
                        return handler(points, language)
                except (ValueError, IndexError):
                    continue
        
        # Patterns belote/rebelote
        belote_patterns = {
            'fr': [
                r'belote.*rebelote',
                r'quand.*utiliser.*belote',
                r'comment.*belote',
                r'roi.*dame.*atout'
            ],
            'en': [
                r'belote.*rebelote',
                r'when.*use.*belote',
                r'how.*belote',
                r'king.*queen.*trump'
            ]
        }
        
        patterns = belote_patterns.get(language, belote_patterns['fr'])
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return self.get_belote_rebelote_info(language)
        
        return None
    
    def get_belote_rebelote_info(self, language: str = 'fr') -> str:
        """Informations officielles belote/rebelote"""
        rule = self.rules_db.get_all_rules()['belote_rebelote_official']
        title = rule['title_fr'] if language == 'fr' else rule['title_en']
        content = rule['content_fr'] if language == 'fr' else rule['content_en']
        return f"**{title}**\n\n{content}"
    
    def handle_hand_evaluation(self, query: str, language: str = 'fr') -> str:
        """Évaluation de main avec recommandations officielles"""
        evaluation = self.hand_evaluator.evaluate_hand_simple(query, language)
        
        if language == 'fr':
            response = f"""**🎯 Analyse de votre main par Sofiene**

**Recommandation officielle:** {evaluation.recommended_announcement} points
**Niveau de confiance:** {evaluation.confidence:.0%}

**Analyse:**
{evaluation.reasoning}

**Alternatives possibles:** {', '.join(map(str, evaluation.alternative_options))} points

**Conseil d'expert:** Vérifiez que votre main respecte les critères officiels pour l'annonce choisie."""
        else:
            response = f"""**🎯 Sofiene's hand analysis**

**Official recommendation:** {evaluation.recommended_announcement} points
**Confidence level:** {evaluation.confidence:.0%}

**Analysis:**
{evaluation.reasoning}

**Possible alternatives:** {', '.join(map(str, evaluation.alternative_options))} points

**Expert advice:** Verify your hand meets official criteria for chosen announcement."""
        
        return response
    
    def get_announcement_recommendation(self, points: int, language: str = 'fr') -> str:
        """Recommandations officielles par niveau d'annonce"""
        recommendations = {
            'fr': {
                90: """**📢 Recommandation officielle pour 90 points**

**Critère obligatoire:** 2 As minimum

**Configuration requise:**
• Main relativement faible mais jouable
• Au moins 2 As dans votre jeu
• Stratégie défensive acceptable

**Exemple de main conforme:**
As♠ As♥ + 6 autres cartes diverses

**Note Sofiene:** Annonce sûre avec cette configuration minimale.""",
                
                100: """**📢 Recommandation officielle pour 100 points**

**Critère officiel:** "Généralement comme tu veux"

**Configuration requise:**
• Flexibilité maximale dans la composition
• Main équilibrée recommandée
• Quelques atouts appréciés

**Exemple de main conforme:**
Composition libre avec équilibre

**Note Sofiene:** Annonce flexible, idéale pour s'adapter au jeu.""",
                
                110: """**📢 Recommandation officielle pour 110 points**

**CRITÈRE OBLIGATOIRE:** Atouts Complets

**Configuration strictement requise:**
• Être sûr de collecter toutes les cartes d'atout dès le début
• **Option 1:** (Valet, 9, As, 10) ou plus
• **Option 2:** (Valet, 9, As, 2+ autres cartes d'atout)

**Exemple de main conforme:**
Valet♠ 9♠ As♠ 10♠ + 4 autres cartes

**⚠️ Attention:** Sans atouts complets, échec quasi-certain!""",
                
                120: """**📢 Recommandation officielle pour 120 points**

**CRITÈRE OBLIGATOIRE:** Maximum 3 couleurs + Atouts Complets

**Configuration strictement requise:**
• Seulement 3 couleurs dans votre main (cœurs, trèfle, carreau)
• Plus atouts complets d'une de ces couleurs

**Cas particulier autorisé:**
• 6 cartes d'atout (dont Valet + 9)
• + 2 autres cartes de couleurs différentes
• Pour avoir exactement 3 couleurs à la main

**Exemple de main conforme:**
Valet♠ 9♠ As♠ 10♠ R♠ D♠ + As♥ + 10♦

**Note Sofiene:** Respectez strictement la limite de 3 couleurs!""",
                
                130: """**📢 Recommandation officielle pour 130 points**

**CRITÈRE OBLIGATOIRE:** Maximum 2 couleurs + Atouts Complets

**Configuration strictement requise:**
• Seulement 2 couleurs dans votre main (cœurs, trèfle, carreau)
• Plus atouts complets d'une de ces couleurs

**Cas particulier autorisé:**
• 6 cartes d'atout (dont Valet + 9)
• + 2 cartes même couleur ≠ atout
• Pour avoir exactement 2 couleurs à la main

**Exemple de main conforme:**
Valet♠ 9♠ As♠ 10♠ R♠ D♠ + As♥ + 10♥

**Note Sofiene:** Configuration très restrictive, soyez certain!""",
                
                140: """**📢 Recommandation officielle pour 140 points**

**CRITÈRE EXTRÊME:** L'adversaire ne peut avoir qu'un seul pli maximum

**Configuration exceptionnelle requise:**
• Main quasi-parfaite obligatoire
• Domination totale du jeu
• Quasi-certitude de tous les plis

**⚠️ TRÈS RISQUÉ**
Réservé aux mains extraordinaires uniquement!

**Note Sofiene:** Annonce exceptionnelle, évaluez avec extrême prudence."""
            },
            'en': {
                90: """**📢 Official recommendation for 90 points**

**Mandatory criterion:** Minimum 2 Aces

**Required configuration:**
• Relatively weak but playable hand
• At least 2 Aces in your game
• Defensive strategy acceptable

**Compliant hand example:**
Ace♠ Ace♥ + 6 other various cards

**Sofiene note:** Safe announcement with this minimal configuration.""",
                
                100: """**📢 Official recommendation for 100 points**

**Official criterion:** "Generally as you wish"

**Required configuration:**
• Maximum flexibility in composition
• Balanced hand recommended
• Some trumps appreciated

**Compliant hand example:**
Free composition with balance

**Sofiene note:** Flexible announcement, ideal for adapting to game.""",
                
                110: """**📢 Official recommendation for 110 points**

**MANDATORY CRITERION:** Complete Trumps

**Strictly required configuration:**
• Must be sure to collect all trump cards from start
• **Option 1:** (Jack, 9, Ace, 10) or more
• **Option 2:** (Jack, 9, Ace, 2+ other trump cards)

**Compliant hand example:**
Jack♠ 9♠ Ace♠ 10♠ + 4 other cards

**⚠️ Warning:** Without complete trumps, almost certain failure!""",
                
                120: """**📢 Official recommendation for 120 points**

**MANDATORY CRITERION:** Maximum 3 colors + Complete Trumps

**Strictly required configuration:**
• Only 3 colors in your hand (hearts, clubs, diamonds)
• Plus complete trumps of one of these colors

**Authorized special case:**
• 6 trump cards (including Jack + 9)
• + 2 other cards of different colors
• To have exactly 3 colors in hand

**Compliant hand example:**
Jack♠ 9♠ Ace♠ 10♠ King♠ Queen♠ + Ace♥ + 10♦

**Sofiene note:** Strictly respect the 3-color limit!""",
                
                130: """**📢 Official recommendation for 130 points**

**MANDATORY CRITERION:** Maximum 2 colors + Complete Trumps

**Strictly required configuration:**
• Only 2 colors in your hand (hearts, clubs, diamonds)
• Plus complete trumps of one of these colors

**Authorized special case:**
• 6 trump cards (including Jack + 9)
• + 2 cards same color ≠ trump
• To have exactly 2 colors in hand

**Compliant hand example:**
Jack♠ 9♠ Ace♠ 10♠ King♠ Queen♠ + Ace♥ + 10♥

**Sofiene note:** Very restrictive configuration, be certain!""",
                
                140: """**📢 Official recommendation for 140 points**

**EXTREME CRITERION:** Opponent can have maximum one trick

**Exceptional configuration required:**
• Near-perfect hand mandatory
• Total game domination
• Near-certainty of all tricks

**⚠️ VERY RISKY**
Reserved for extraordinary hands only!

**Sofiene note:** Exceptional announcement, evaluate with extreme caution."""
            }
        }
        
        return recommendations.get(language, recommendations['fr']).get(points, 
            f"Aucune recommandation pour {points} points." if language == 'fr' 
            else f"No recommendation for {points} points.")
    
    def get_announcement_conditions(self, points: int, language: str = 'fr') -> str:
        """Conditions officielles pour chaque niveau d'annonce"""
        conditions = {
            'fr': {
                90: "**Quand annoncer 90 points:**\n• Avec au moins 2 As\n• Main faible mais jouable\n• Stratégie défensive",
                100: "**Quand annoncer 100 points:**\n• \"Généralement comme tu veux\"\n• Main équilibrée\n• Flexibilité maximale",
                110: "**Quand annoncer 110 points:**\n• SEULEMENT avec atouts complets\n• (V-9-A-10) minimum requis\n• Confiance totale de collecter atouts",
                120: "**Quand annoncer 120 points:**\n• Maximum 3 couleurs + atouts complets\n• Configuration stricte obligatoire\n• Vérifier critères officiels",
                130: "**Quand annoncer 130 points:**\n• Maximum 2 couleurs + atouts complets\n• Configuration très restrictive\n• Évaluation précise nécessaire",
                140: "**Quand annoncer 140 points:**\n• Main exceptionnelle uniquement\n• Adversaire max 1 pli\n• Risque extrême!"
            },
            'en': {
                90: "**When to announce 90 points:**\n• With at least 2 Aces\n• Weak but playable hand\n• Defensive strategy",
                100: "**When to announce 100 points:**\n• \"Generally as you wish\"\n• Balanced hand\n• Maximum flexibility",
                110: "**When to announce 110 points:**\n• ONLY with complete trumps\n• (J-9-A-10) minimum required\n• Total confidence to collect trumps",
                120: "**When to announce 120 points:**\n• Maximum 3 colors + complete trumps\n• Strict configuration mandatory\n• Verify official criteria",
                130: "**When to announce 130 points:**\n• Maximum 2 colors + complete trumps\n• Very restrictive configuration\n• Precise evaluation needed",
                140: "**When to announce 140 points:**\n• Exceptional hand only\n• Opponent max 1 trick\n• Extreme risk!"
            }
        }
        
        return conditions.get(language, conditions['fr']).get(points, 
            f"Conditions pour {points} points non définies." if language == 'fr' 
            else f"Conditions for {points} points not defined.")
            
    def find_best_matches(self, query: str, language: str = 'fr', top_k: int = 3) -> List[RuleMatch]:
        """Trouver les meilleures correspondances avec boost de mots-clés"""
        if not self.model or not self.rule_embeddings:
            return []
            
        try:
            query_embedding = self.model.encode(query)
            matches = []
            
            # Boost de mots-clés spécialisés
            boost_keywords = {
                'fr': {
                    'belote': 0.6, 'rebelote': 0.6, 'roi': 0.4, 'dame': 0.4,
                    'recommandation': 0.5, 'annoncer': 0.5, 'points': 0.3,
                    'utiliser': 0.4, 'quand': 0.3, 'officiel': 0.4,
                    '110': 0.6, '120': 0.6, '130': 0.6, '140': 0.6, '90': 0.5, '100': 0.5
                },
                'en': {
                    'belote': 0.6, 'rebelote': 0.6, 'king': 0.4, 'queen': 0.4,
                    'recommendation': 0.5, 'announce': 0.5, 'points': 0.3,
                    'use': 0.4, 'when': 0.3, 'official': 0.4,
                    '110': 0.6, '120': 0.6, '130': 0.6, '140': 0.6, '90': 0.5, '100': 0.5
                }
            }
            
            for rule_id, rule_data in self.rule_embeddings.items():
                rule_embedding = rule_data[language]
                similarity = cosine_similarity([query_embedding], [rule_embedding])[0][0]
                
                # Appliquer le boost de mots-clés
                query_lower = query.lower()
                boost_amount = 0
                for keyword, boost in boost_keywords.get(language, {}).items():
                    if keyword in query_lower:
                        boost_amount += boost
                
                similarity += min(boost_amount, 0.9)  # Plafond du boost
                
                # Boost pour correspondances de patterns
                rule = rule_data['rule']
                if 'patterns_' + language in rule:
                    for pattern in rule['patterns_' + language]:
                        if re.search(pattern, query.lower()):
                            similarity += 0.5  # Boost pattern
                            break
                
                matches.append(RuleMatch(
                    rule_id=rule_id,
                    score=similarity,
                    rule_data=rule_data['rule']
                ))
                
            matches.sort(key=lambda x: x.score, reverse=True)
            return matches[:top_k]
        except Exception as e:
            st.error(f"Erreur de recherche: {str(e)}")
            return []
        
    def extract_intent(self, query: str, language: str = 'fr') -> str:
        """Extraire l'intention de la requête"""
        query_lower = query.lower()
        
        intent_keywords = {
            'fr': {
                'belote_rebelote': ['belote', 'rebelote', 'roi.*dame', 'bonus.*20'],
                'hand_evaluation': ['j\'ai', 'main', 'que.*annoncer', 'évaluer', 'analyser'],
                'announcements': ['recommandation', 'annoncer', 'contrat', '90', '100', '110', '120', '130', '140'],
                'scoring': ['point', 'score', 'calcul', 'comptage'],
                'cards': ['carte', 'valeur', 'atout', 'couleur'],
                'coinche': ['coinche', 'surcoinche', 'multiplicateur'],
                'strategy': ['stratégie', 'conseil', 'astuce', 'tactique'],
                'basic': ['règle', 'base', 'comment', 'début', 'jeu'],
                'capot': ['capot', 'tous', 'plis']
            },
            'en': {
                'belote_rebelote': ['belote', 'rebelote', 'king.*queen', 'bonus.*20'],
                'hand_evaluation': ['i have', 'hand', 'what.*announce', 'evaluate', 'analyze'],
                'announcements': ['recommendation', 'announce', 'contract', '90', '100', '110', '120', '130', '140'],
                'scoring': ['point', 'score', 'calculate', 'counting'],
                'cards': ['card', 'value', 'trump', 'color'],
                'coinche': ['coinche', 'surcoinche', 'multiplier'],
                'strategy': ['strategy', 'advice', 'tip', 'tactic'],
                'basic': ['rule', 'basic', 'how', 'start', 'game'],
                'capot': ['capot', 'all', 'tricks']
            }
        }
        
        keywords = intent_keywords.get(language, intent_keywords['fr'])
        
        # Prioriser belote/rebelote
        for word in keywords['belote_rebelote']:
            if re.search(word, query_lower):
                return 'belote_rebelote'
        
        # Vérifier évaluation de main
        for word in keywords['hand_evaluation']:
            if re.search(word, query_lower):
                return 'hand_evaluation'
        
        # Vérifier autres intentions
        for intent, words in keywords.items():
            if intent in ['belote_rebelote', 'hand_evaluation']:
                continue
            for word in words:
                if re.search(word, query_lower):
                    return intent
                    
        return 'general'
        
    def generate_contextual_response(self, matches: List[RuleMatch], intent: str, 
                                   language: str = 'fr', context: List[str] = None) -> str:
        """Générer une réponse contextuelle"""
        if not matches or matches[0].score < 0.3:
            return self.get_fallback_response(intent, language, context)
            
        best_match = matches[0]
        rule = best_match.rule_data
        
        title = rule['title_fr'] if language == 'fr' else rule['title_en']
        content = rule['content_fr'] if language == 'fr' else rule['content_en']
        
        response = f"**{title}**\n\n{content}"
        
        # Ajouter conseil d'expert pour annonces
        if intent == 'announcements' and best_match.score > 0.7:
            expert_tip = "\n\n**💡 Conseil d'expert Sofiene:**\n• Respectez strictement les critères officiels\n• En cas de doute, optez pour une annonce plus conservatrice\n• Observez attentivement le jeu de vos adversaires" if language == 'fr' else "\n\n**💡 Sofiene expert tip:**\n• Strictly follow official criteria\n• When in doubt, choose more conservative announcement\n• Carefully observe your opponents' game"
            response += expert_tip
                
        if best_match.score > 0.8 and len(matches) > 1:
            related_header = "**Voir aussi:**" if language == 'fr' else "**See also:**"
            response += f"\n\n{related_header}\n"
            for match in matches[1:2]:
                related_title = match.rule_data['title_fr'] if language == 'fr' else match.rule_data['title_en']
                response += f"• {related_title}\n"
                
        return response
        
    def get_fallback_response(self, intent: str, language: str = 'fr', context: List[str] = None) -> str:
        """Réponse de secours quand aucune correspondance n'est trouvée"""
        fallbacks = {
            'fr': {
                'belote_rebelote': "La Belote et Rebelote sont définies par avoir le Roi ET la Dame d'atout chez le même joueur. Annoncez 'Belote' puis 'Rebelote' en jouant ces cartes pour obtenir +20 points à votre équipe.",
                'hand_evaluation': "Pour évaluer votre main selon les règles officielles, décrivez-moi vos cartes en détail. Par exemple: 'J'ai Valet, 9, As et 10 de carreau plus 4 autres cartes.' Je vous donnerai la recommandation officielle appropriée.",
                'announcements': "Je peux vous expliquer les recommandations officielles pour chaque niveau d'annonce (90, 100, 110, 120, 130, 140). Quel niveau vous intéresse?",
                'scoring': "Le système de score officiel de la Belote Contrée suit des règles précises. Voulez-vous connaître le calcul des points, le système de contrats, ou les bonus?",
                'cards': "Les cartes ont des valeurs officielles différentes à l'atout et hors atout. Voulez-vous connaître les valeurs spécifiques et l'ordre de force?",
                'coinche': "Le système Coinche officiel multiplie les gains et risques (×1, ×2, ×4). Voulez-vous en savoir plus sur les multiplicateurs?",
                'strategy': "Je peux partager des stratégies officielles et conseils d'expert pour améliorer votre jeu. Quel aspect vous intéresse?",
                'basic': "Je peux expliquer les règles officielles de base de la Belote Contrée. Par quoi voulez-vous commencer?",
                'general': "Je suis Sofiene, votre expert en Belote Tunisienne Contrée. Je peux vous aider avec les règles officielles, les annonces, l'évaluation de main, le scoring, ou les stratégies. Que souhaitez-vous savoir?"
            },
            'en': {
                'belote_rebelote': "Belote and Rebelote are defined by having King AND Queen of trump with the same player. Announce 'Belote' then 'Rebelote' when playing these cards to get +20 points for your team.",
                'hand_evaluation': "To evaluate your hand according to official rules, describe your cards in detail. For example: 'I have Jack, 9, Ace and 10 of diamonds plus 4 other cards.' I'll give you the appropriate official recommendation.",
                'announcements': "I can explain official recommendations for each announcement level (90, 100, 110, 120, 130, 140). Which level interests you?",
                'scoring': "The official Belote Contrée scoring system follows precise rules. Would you like to know about point calculation, contract system, or bonuses?",
                'cards': "Cards have different official values for trump and non-trump. Would you like to know specific values and strength order?",
                'coinche': "The official Coinche system multiplies gains and risks (×1, ×2, ×4). Would you like to know more about multipliers?",
                'strategy': "I can share official strategies and expert tips to improve your game. What aspect interests you?",
                'basic': "I can explain official basic rules of Belote Contrée. Where would you like to start?",
                'general': "I'm Sofiene, your Tunisian Belote Contrée expert. I can help with official rules, announcements, hand evaluation, scoring, or strategies. What would you like to know?"
            }
        }
        
        return fallbacks.get(language, fallbacks['fr']).get(intent, fallbacks[language]['general'])
        
    def process_query(self, query: str, language: str = 'fr', context: List[str] = None) -> str:
        """Méthode principale pour traiter les requêtes"""
        # Essayer d'abord la reconnaissance de patterns spécifiques
        pattern_response = self.handle_specific_patterns(query, language)
        if pattern_response:
            return pattern_response
        
        # Recherche sémantique en secours
        intent = self.extract_intent(query, language)
        matches = self.find_best_matches(query, language)
        response = self.generate_contextual_response(matches, intent, language, context)
        return response

def init_session_state():
    """Initialiser l'état de session Streamlit"""
    if 'conversation' not in st.session_state:
        st.session_state.conversation = ConversationManager()
    if 'ai' not in st.session_state:
        st.session_state.ai = SofieneAI()
        st.session_state.ai.rule_embeddings = st.session_state.ai.initialize_embeddings()
    if 'language' not in st.session_state:
        st.session_state.language = 'fr'
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def get_quick_suggestions(language: str):
    """Suggestions rapides basées sur la langue"""
    if language == 'fr':
        return [
            "Recommandation pour 120 points",
            "Quand utiliser belote rebelote?",
            "Recommandation pour 110 points",
            "J'ai Valet, 9, As et 10 carreau, que dois-je annoncer?",
            "Valeurs officielles des cartes",
            "Comment calculer les points?",
            "Règles du capot"
        ]
    else:
        return [
            "Recommendation for 120 points",
            "When to use belote rebelote?",
            "Recommendation for 110 points",
            "I have Jack, 9, Ace and 10 diamonds, what should I announce?",
            "Official card values",
            "How to calculate points?",
            "Capot rules"
        ]

def process_message(message: str):
    """Traiter un message et ajouter la réponse au chat"""
    st.session_state.messages.append({"role": "user", "content": message})
    st.session_state.conversation.add_message("user", message)
    
    context = st.session_state.conversation.get_context()
    response = st.session_state.ai.process_query(message, st.session_state.language, context)
    
    st.session_state.conversation.add_message("bot", response)
    st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    """Application Streamlit principale - Production Ready"""
    
    st.set_page_config(
        page_title="Sofiene - Expert Belote Tunisienne",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        margin-bottom: 5px;
    }
    .sofiene-header {
        background: linear-gradient(90deg, #1f4e79, #2d5aa0);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .expert-badge {
        background-color: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    .footer-dev {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 1rem;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("""
        <div class="sofiene-header">
            <h1>🎮 Sofiene</h1>
            <p>Expert en Belote Tunisienne Contrée</p>
            <span class="expert-badge">Règles Officielles</span>
        </div>
        """, unsafe_allow_html=True)
        
        current_lang = st.session_state.language
        if st.button(f"🌍 {'Français' if current_lang == 'en' else 'English'}"):
            st.session_state.language = 'en' if current_lang == 'fr' else 'fr'
            st.rerun()
        
        st.divider()
        
        suggestions_title = "Questions suggérées:" if st.session_state.language == 'fr' else "Suggested questions:"
        st.subheader(suggestions_title)
        
        suggestions = get_quick_suggestions(st.session_state.language)
        for i, suggestion in enumerate(suggestions):
            if st.button(suggestion, key=f"suggestion_{i}_{st.session_state.language}"):
                process_message(suggestion)
                st.rerun()
        
        st.divider()
        
        # Référence rapide officielle
        if st.session_state.language == 'fr':
            st.subheader("📋 Référence Officielle")
            st.markdown("""
            **Annonces:**
            • 90: 2 As minimum
            • 100: "Comme tu veux"
            • 110: Atouts complets
            • 120: Max 3 couleurs + atouts
            • 130: Max 2 couleurs + atouts
            • 140: Adversaire max 1 pli
            
            **Belote/Rebelote:**
            • Roi + Dame d'atout = +20 pts
            
            **Total par manche:**
            • 152 + 10 (dix de der) = 162 pts
            """)
        else:
            st.subheader("📋 Official Reference")
            st.markdown("""
            **Announcements:**
            • 90: 2 Aces minimum
            • 100: "As you wish"
            • 110: Complete trumps
            • 120: Max 3 colors + trumps
            • 130: Max 2 colors + trumps
            • 140: Opponent max 1 trick
            
            **Belote/Rebelote:**
            • King + Queen trump = +20 pts
            
            **Total per round:**
            • 152 + 10 (ten of last) = 162 pts
            """)
        
        st.divider()
        
        if st.button("💾 " + ("Exporter" if st.session_state.language == 'fr' else "Export")):
            filename = f"sofiene_conversation_{st.session_state.conversation.get_timestamp()}.txt"
            if st.session_state.conversation.export_to_file(filename, st.session_state.language):
                st.success(f"Exporté vers {filename}" if st.session_state.language == 'fr' else f"Exported to {filename}")
                
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        st.download_button(
                            label="📥 " + ("Télécharger" if st.session_state.language == 'fr' else "Download"),
                            data=f.read(),
                            file_name=filename,
                            mime="text/plain"
                        )
                except Exception as e:
                    st.warning(f"Impossible de créer le lien de téléchargement: {str(e)}" if st.session_state.language == 'fr' 
                              else f"Cannot create download link: {str(e)}")
        
        summary = st.session_state.conversation.get_conversation_summary(st.session_state.language)
        st.info(summary)
        
        # Footer développeur
        st.markdown("""
        <div class="footer-dev">
            <p>🚀 Développé par <strong>BellaajMohsen7</strong></p>
            <p>Expert en Belote Tunisienne</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Zone de contenu principal
    if st.session_state.language == 'fr':
        st.title("🎮 Sofiene - Expert en Belote Tunisienne Contrée")
        st.markdown("""
        **Votre assistant personnel pour maîtriser la Belote Contrée**
        
        Sofiene vous aide avec les règles officielles, les recommandations d'annonces précises, 
        l'évaluation de main et toutes les stratégies avancées de la Belote Tunisienne.
        
        **🎯 Expertise disponible:**
        • Recommandations officielles pour chaque niveau d'annonce
        • Évaluation précise de vos mains
        • Règles complètes Belote/Rebelote
        • Système de scoring officiel
        • Conseils stratégiques d'expert
        """)
    else:
        st.title("🎮 Sofiene - Tunisian Belote Contrée Expert")
        st.markdown("""
        **Your personal assistant to master Belote Contrée**
        
        Sofiene helps you with official rules, precise announcement recommendations, 
        hand evaluation and all advanced strategies of Tunisian Belote.
        
        **🎯 Available expertise:**
        • Official recommendations for each announcement level
        • Precise evaluation of your hands
        • Complete Belote/Rebelote rules
        • Official scoring system
        • Expert strategic advice
        """)
    
    # Section de test rapide
    if st.session_state.language == 'fr':
        with st.expander("🧪 Testez l'expertise de Sofiene"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Questions d'annonces:**
                • "Recommandation pour 120 points"
                • "Quand annoncer 110 points?"
                • "Critères pour 130 points"
                """)
            
            with col2:
                st.markdown("""
                **Questions techniques:**
                • "Quand utiliser belote rebelote?"
                • "Comment calculer les points?"
                • "Valeurs des cartes à l'atout"
                """)
    else:
        with st.expander("🧪 Test Sofiene's expertise"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Announcement questions:**
                • "Recommendation for 120 points"
                • "When to announce 110 points?"
                • "Criteria for 130 points"
                """)
            
            with col2:
                st.markdown("""
                **Technical questions:**
                • "When to use belote rebelote?"
                • "How to calculate points?"
                • "Trump card values"
                """)
    
    # Interface de chat
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Zone de saisie
    prompt_text = "Posez votre question sur la Belote..." if st.session_state.language == 'fr' else "Ask your Belote question..."
    
    if prompt := st.chat_input(prompt_text):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyse par Sofiene..." if st.session_state.language == 'fr' else "Sofiene analyzing..."):
                try:
                    process_message(prompt)
                    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                        st.markdown(st.session_state.messages[-1]["content"])
                        st.rerun()
                        
                except Exception as e:
                    error_msg = f"Erreur d'analyse: {str(e)}" if st.session_state.language == 'fr' else f"Analysis error: {str(e)}"
                    st.error(error_msg)
                    
                    # Message de fallback en cas d'erreur
                    fallback_msg = "Je rencontre une difficulté technique. Reformulez votre question ou essayez une question plus simple." if st.session_state.language == 'fr' else "I'm experiencing a technical difficulty. Please rephrase your question or try a simpler one."
                    st.markdown(fallback_msg)
    
    # Footer principal avec informations
    st.divider()
    
    if st.session_state.language == 'fr':
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🎯 Spécialités**
            • Annonces officielles
            • Évaluation de main
            • Stratégies avancées
            """)
        
        with col2:
            st.markdown("""
            **📚 Base de connaissances**
            • Règles officielles
            • Calculs de score
            • Belote/Rebelote
            """)
        
        with col3:
            st.markdown("""
            **💡 Conseils d'expert**
            • Recommandations précises
            • Analyses détaillées
            • Stratégies gagnantes
            """)
        
        st.markdown("""
        ---
        **🚀 Développé avec passion par BellaajMohsen7**  
        *Votre expert en Belote Tunisienne Contrée - Toujours prêt à vous conseiller!*
        
        📧 Contact: BellaajMohsen7@github.com | 🌟 Version 1.0 Production
        """)
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🎯 Specialties**
            • Official announcements
            • Hand evaluation
            • Advanced strategies
            """)
        
        with col2:
            st.markdown("""
            **📚 Knowledge base**
            • Official rules
            • Score calculations
            • Belote/Rebelote
            """)
        
        with col3:
            st.markdown("""
            **💡 Expert advice**
            • Precise recommendations
            • Detailed analysis
            • Winning strategies
            """)
        
        st.markdown("""
        ---
        **🚀 Developed with passion by BellaajMohsen7**  
        *Your Tunisian Belote Contrée expert - Always ready to advise you!*
        
        📧 Contact: BellaajMohsen7@github.com | 🌟 Version 1.0 Production
        """)

if __name__ == "__main__":
    main()