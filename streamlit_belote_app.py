#!/usr/bin/env python3
"""
Tunisian Belote Rules Chatbot - Streamlit Web Application
A sophisticated web-based chatbot for learning Belote rules
"""

import streamlit as st
import numpy as np
import pickle
import os
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

# Try to import required libraries with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    st.error("Please install required dependencies: pip install sentence-transformers scikit-learn")

@dataclass
class RuleMatch:
    rule_id: str
    score: float
    rule_data: Dict

class BeloteRulesDatabase:
    """Comprehensive database of Tunisian Belote rules"""
    
    def __init__(self):
        self.rules = self._initialize_rules()
        
    def _initialize_rules(self):
        """Initialize the complete rules database"""
        return {
            'basic_rules': {
                'id': 'basic_rules',
                'category': 'basic',
                'title_fr': '🎮 Règles de Base',
                'title_en': '🎮 Basic Rules',
                'content_fr': """**Configuration du jeu:**
• 4 joueurs organisés en 2 équipes de 2
• Jeu de 32 cartes (du 7 à l'As)
• Distribution de 8 cartes par joueur

**Déroulement d'un tour:**
• Le joueur à droite du donneur commence
• Obligation de suivre la couleur demandée
• Si impossible, peut jouer n'importe quelle carte
• Le pli est remporté par la carte la plus forte ou l'atout le plus fort

**Annonce de l'atout:**
• Chaque joueur peut annoncer un contrat
• L'annonce la plus forte détermine l'atout
• Si personne n'annonce, redistribution des cartes""",
                'content_en': """**Game setup:**
• 4 players organized in 2 teams of 2
• 32-card deck (7 to Ace)
• 8 cards dealt per player

**Turn progression:**
• Player to the right of dealer starts
• Must follow the requested suit
• If impossible, can play any card
• Trick won by highest card or strongest trump

**Trump announcement:**
• Each player can announce a contract
• Highest announcement determines trump
• If no one announces, cards are redistributed""",
                'keywords_fr': ['règles', 'base', 'joueurs', 'cartes', 'distribution', 'tour', 'pli', 'atout', 'annonce', 'équipe', 'jeu', 'début', 'comment'],
                'keywords_en': ['rules', 'basic', 'players', 'cards', 'distribution', 'turn', 'trick', 'trump', 'announcement', 'team', 'game', 'start', 'how'],
                'examples_fr': [
                    'Si on joue Cœur et que vous n\'avez pas de Cœur, vous pouvez couper avec un atout',
                    'Le joueur qui a annoncé le contrat le plus élevé choisit l\'atout'
                ],
                'examples_en': [
                    'If Hearts is played and you have no Hearts, you can cut with a trump',
                    'The player who announced the highest contract chooses trump'
                ]
            },
            
            'card_values': {
                'id': 'card_values',
                'category': 'cards',
                'title_fr': '🃏 Valeurs des Cartes',
                'title_en': '🃏 Card Values',
                'content_fr': """**À l'atout:**
• Valet: 20 points 👑 (carte la plus forte)
• 9: 14 points
• As: 11 points  
• 10: 10 points
• Roi: 4 points
• Dame: 3 points
• 8, 7: 0 point

**Hors atout:**
• As: 11 points 👑 (carte la plus forte)
• 10: 10 points
• Roi: 4 points
• Dame: 3 points
• Valet: 2 points
• 9, 8, 7: 0 point

**Total possible par manche:** 152 points + 10 points (dix de der) = 162 points""",
                'content_en': """**Trump cards:**
• Jack: 20 points 👑 (strongest card)
• 9: 14 points
• Ace: 11 points
• 10: 10 points
• King: 4 points
• Queen: 3 points
• 8, 7: 0 points

**Non-trump cards:**
• Ace: 11 points 👑 (strongest card)
• 10: 10 points
• King: 4 points
• Queen: 3 points
• Jack: 2 points
• 9, 8, 7: 0 points

**Total possible per round:** 152 points + 10 points (ten of last) = 162 points""",
                'keywords_fr': ['valeurs', 'cartes', 'atout', 'valet', 'as', 'points', 'roi', 'dame', 'carte', 'valeur', 'couleur'],
                'keywords_en': ['values', 'cards', 'trump', 'jack', 'ace', 'points', 'king', 'queen', 'card', 'value', 'color'],
                'examples_fr': [
                    'Le Valet d\'atout bat toutes les autres cartes, même un As d\'atout',
                    'Un As de Pique (hors atout) vaut 11 points, plus qu\'un Roi d\'atout (4 points)'
                ],
                'examples_en': [
                    'Trump Jack beats all other cards, even trump Ace',
                    'Ace of Spades (non-trump) worth 11 points, more than trump King (4 points)'
                ]
            },
            
            'scoring_system': {
                'id': 'scoring_system',
                'category': 'scoring',
                'title_fr': '📊 Système de Score',
                'title_en': '📊 Scoring System',
                'content_fr': """**Calcul des points:**
• Points des cartes: 152 maximum
• Dix de der (dernier pli): +10 points
• **Total possible: 162 points**

**Système de score spécial:**
Pour l'équipe non-preneuse, si score = 10×K + x:
• Si x ∈ [5,6,7] → Score final = 10×(K+1)
• Sinon → Score final = 10×K

**Bonus:**
• Belote + Rebelote: +20 points (Roi et Dame d'atout dans la même main)

**Échec de contrat:**
• Équipe preneuse: 0 points
• Équipe adverse: 160 + 20×(bonus belote)

**Fin de partie:** Premier à atteindre 1000 ou 2000 points""",
                'content_en': """**Point calculation:**
• Card points: 152 maximum
• Ten of last (last trick): +10 points
• **Total possible: 162 points**

**Special scoring system:**
For non-taking team, if score = 10×K + x:
• If x ∈ [5,6,7] → Final score = 10×(K+1)
• Otherwise → Final score = 10×K

**Bonus:**
• Belote + Rebelote: +20 points (King and Queen of trump in same hand)

**Contract failure:**
• Taking team: 0 points
• Opposing team: 160 + 20×(belote bonus)

**Game end:** First to reach 1000 or 2000 points""",
                'keywords_fr': ['score', 'points', 'calcul', 'belote', 'rebelote', 'échec', 'contrat', 'point', 'comptage'],
                'keywords_en': ['score', 'points', 'calculation', 'belote', 'rebelote', 'failure', 'contract', 'point', 'counting'],
                'examples_fr': [
                    'Équipe fait 67 points → arrondi à 70, l\'autre équipe obtient 90',
                    'Contrat 120 échoué: preneur 0 points, adversaire 160 points'
                ],
                'examples_en': [
                    'Team scores 67 points → rounded to 70, other team gets 90',
                    '120 contract failed: taker 0 points, opponent 160 points'
                ]
            },
            
            'announcements': {
                'id': 'announcements',
                'category': 'announcements',
                'title_fr': '📢 Système d\'Annonces',
                'title_en': '📢 Announcement System',
                'content_fr': """**Guide des annonces par niveau:**

**90 points:** Minimum 2 As recommandés
• Main relativement faible mais jouable

**100 points:** Composition flexible
• "Comme tu veux" - assez de liberté

**110 points:** Atouts complets requis
• Minimum: Valet + 9 + As + 10 d'atout
• Ou plus de cartes d'atout fortes

**120 points:** Maximum 3 couleurs + atouts complets
• Cas spécial: 6 atouts (dont Valet + 9) + 2 autres cartes

**130 points:** Maximum 2 couleurs + atouts complets
• Cas spécial: 6 atouts (dont Valet + 9) + 2 cartes même couleur

**140 points:** Quasi-capot
• L'adversaire ne peut faire qu'un seul pli maximum""",
                'content_en': """**Announcement guide by level:**

**90 points:** Minimum 2 Aces recommended
• Relatively weak but playable hand

**100 points:** Flexible composition
• "As you wish" - quite flexible

**110 points:** Complete trumps required
• Minimum: Jack + 9 + Ace + 10 of trump
• Or more strong trump cards

**120 points:** Maximum 3 colors + complete trumps
• Special case: 6 trumps (including Jack + 9) + 2 other cards

**130 points:** Maximum 2 colors + complete trumps
• Special case: 6 trumps (including Jack + 9) + 2 cards same color

**140 points:** Near-capot
• Opponent can make maximum one trick""",
                'keywords_fr': ['annonce', 'contrat', '90', '100', '110', '120', '130', '140', 'atouts', 'recommandations'],
                'keywords_en': ['announcement', 'contract', '90', '100', '110', '120', '130', '140', 'trumps', 'recommendations'],
                'examples_fr': [
                    'Pour 110: Valet♠ 9♠ As♠ 10♠ Roi♠ + quelques cartes hors atout',
                    'Pour 140: Main exceptionnelle, presque sûr de faire tous les plis'
                ],
                'examples_en': [
                    'For 110: Jack♠ 9♠ Ace♠ 10♠ King♠ + some non-trump cards',
                    'For 140: Exceptional hand, almost sure to make all tricks'
                ]
            },
            
            'coinche_system': {
                'id': 'coinche_system',
                'category': 'coinche',
                'title_fr': '🎯 Système Coinche',
                'title_en': '🎯 Coinche System',
                'content_fr': """**Multiplicateurs de risque:**
• Contrat simple: ×1 (normal)
• Coinché: ×2 🔥
• Surcoinché: ×4 🔥🔥

**Mécanisme:**
• Coinche: Adversaire pense que le contrat va échouer
• Surcoinche: Preneur maintient sa confiance malgré la coinche

**Calcul en cas d'échec:**
• Contrat simple échoué: adversaire marque les points du contrat
• Contrat coinché échoué: adversaire marque le **double** du contrat
• Contrat surcoinché échoué: adversaire marque le **quadruple** du contrat

**Stratégie:**
• Coincher: Seulement si très confiant de l'échec adversaire
• Surcoincher: Seulement avec une main exceptionnelle""",
                'content_en': """**Risk multipliers:**
• Simple contract: ×1 (normal)
• Coinched: ×2 🔥
• Surcoinched: ×4 🔥🔥

**Mechanism:**
• Coinche: Opponent thinks contract will fail
• Surcoinche: Taker maintains confidence despite coinche

**Calculation on failure:**
• Simple contract failed: opponent scores contract points
• Coinched contract failed: opponent scores **double** the contract
• Surcoinched contract failed: opponent scores **quadruple** the contract

**Strategy:**
• Coinching: Only if very confident of opponent's failure
• Surcoinching: Only with exceptional hand""",
                'keywords_fr': ['coinche', 'surcoinche', 'multiplicateur', 'risque', 'double', 'quadruple'],
                'keywords_en': ['coinche', 'surcoinche', 'multiplier', 'risk', 'double', 'quadruple'],
                'examples_fr': [
                    'Contrat 120 coinché qui échoue: adversaire marque 240 points!',
                    'Surcoincher un 110: risque de donner 440 points à l\'adversaire'
                ],
                'examples_en': [
                    '120 coinched contract that fails: opponent scores 240 points!',
                    'Surcoinching a 110: risk of giving 440 points to opponent'
                ]
            },
            
            'strategy_tips': {
                'id': 'strategy_tips',
                'category': 'strategy',
                'title_fr': '💡 Stratégies et Conseils',
                'title_en': '💡 Strategies and Tips',
                'content_fr': """**Gestion des atouts:**
• Conservez vos atouts pour les plis cruciaux
• Le Valet et le 9 d'atout sont précieux
• Coupez intelligemment pour prendre des points

**Observation et mémoire:**
• Suivez les cartes jouées pour anticiper
• Notez les couleurs manquantes chez les adversaires
• Mémorisez les gros atouts sortis

**Communication en équipe:**
• Annonces: donnent des indices sur votre main
• Cartes jouées: signalez vos forces et faiblesses
• Attention aux signaux adverses

**Tactiques avancées:**
• Expulsez les atouts adverses en début de jeu
• Gardez des cartes maîtresses dans chaque couleur
• Équilibrez attaque et défense selon le contrat""",
                'content_en': """**Trump management:**
• Save your trumps for crucial tricks
• Trump Jack and 9 are precious
• Cut intelligently to take points

**Observation and memory:**
• Track played cards to anticipate
• Note missing colors in opponents
• Remember big trumps that came out

**Team communication:**
• Announcements: give clues about your hand
• Played cards: signal your strengths and weaknesses
• Watch for opponent signals

**Advanced tactics:**
• Expel opponent trumps early in game
• Keep master cards in each color
• Balance attack and defense according to contract""",
                'keywords_fr': ['stratégie', 'conseil', 'tactique', 'atouts', 'équipe', 'observation', 'astuce'],
                'keywords_en': ['strategy', 'advice', 'tactic', 'trumps', 'team', 'observation', 'tip'],
                'examples_fr': [
                    'Jouez vos gros atouts tôt pour forcer les adversaires à sous-couper',
                    'Si partenaire annonce 120, soutenez-le en évitant de prendre ses plis'
                ],
                'examples_en': [
                    'Play your big trumps early to force opponents to under-cut',
                    'If partner announces 120, support them by avoiding taking their tricks'
                ]
            },
            
            'capot_rules': {
                'id': 'capot_rules',
                'category': 'advanced',
                'title_fr': '🏆 Règles du Capot',
                'title_en': '🏆 Capot Rules',
                'content_fr': """**Définition du Capot:**
• Une équipe remporte TOUS les plis (8 plis sur 8)
• Score automatique: 250 points
• Ignore le calcul normal des points de cartes

**Capot dans un contrat:**
• Si annoncé dans le contrat: DOIT réussir tous les plis
• Échec = chute du contrat même avec beaucoup de points

**Capot surprise:**
• Pas annoncé mais réalisé pendant le jeu
• Bonus de 250 points pour l'équipe

**Stratégie Capot:**
• Main exceptionnelle requise
• Minimum 6-7 atouts avec Valet et 9
• Maîtres dans au moins 2 couleurs
• Communication cruciale avec le partenaire""",
                'content_en': """**Capot Definition:**
• One team wins ALL tricks (8 out of 8)
• Automatic score: 250 points
• Ignores normal card point calculation

**Capot in contract:**
• If announced in contract: MUST succeed all tricks
• Failure = contract failure even with many points

**Surprise Capot:**
• Not announced but achieved during play
• 250 points bonus for the team

**Capot Strategy:**
• Exceptional hand required
• Minimum 6-7 trumps with Jack and 9
• Masters in at least 2 colors
• Crucial communication with partner""",
                'keywords_fr': ['capot', 'tous', 'plis', '250', 'points', 'équipe', 'bonus', 'règles'],
                'keywords_en': ['capot', 'all', 'tricks', '250', 'points', 'team', 'bonus', 'rules'],
                'examples_fr': [
                    'Main pour capot: V♠ 9♠ A♠ 10♠ R♠ D♠ + A♥ 10♥',
                    'Capot réussi = 250 points même si cartes valent moins'
                ],
                'examples_en': [
                    'Capot hand: J♠ 9♠ A♠ 10♠ K♠ Q♠ + A♥ 10♥',
                    'Successful capot = 250 points even if cards worth less'
                ]
            }
        }
        
    def get_all_rules(self):
        """Return all rules"""
        return self.rules

class ConversationManager:
    """Conversation management and context handling"""
    
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.context_window = 5
        
    def add_message(self, sender: str, content: str):
        """Add a message to conversation history"""
        message = {
            'sender': sender,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        self.messages.append(message)
        
        if len(self.messages) > self.context_window * 2:
            self.messages = self.messages[-self.context_window * 2:]
            
    def get_context(self) -> List[str]:
        """Get recent conversation context"""
        recent_messages = self.messages[-self.context_window:]
        return [msg['content'] for msg in recent_messages if msg['sender'] == 'user']
        
    def clear_history(self):
        """Clear conversation history"""
        self.messages.clear()
        
    def get_timestamp(self) -> str:
        """Get current timestamp for file naming"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def export_to_file(self, filename: str, language: str = 'fr'):
        """Export conversation to text file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                header = "Conversation Belote Bot" if language == 'fr' else "Belote Bot Conversation"
                f.write(f"=== {header} ===\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for msg in self.messages:
                    sender_label = "Vous" if msg['sender'] == 'user' and language == 'fr' else \
                                  "You" if msg['sender'] == 'user' else \
                                  "Bot"
                    f.write(f"{sender_label}: {msg['content']}\n\n")
            return True
        except Exception as e:
            st.error(f"Error exporting conversation: {str(e)}")
            return False
                
    def get_conversation_summary(self, language: str = 'fr') -> str:
        """Get a summary of the conversation"""
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
    """Load and cache the sentence transformer model"""
    if DEPENDENCIES_AVAILABLE:
        try:
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    return None

class BeloteAI:
    """Advanced AI engine for Tunisian Belote rules"""
    
    def __init__(self):
        self.model = load_sentence_transformer()
        self.rules_db = BeloteRulesDatabase()
        self.rule_embeddings = {}
        self.context_window = 3
        
        if self.model:
            self.initialize_embeddings()
        
    @st.cache_data
    def initialize_embeddings(_self):
        """Initialize or load pre-computed embeddings for all rules"""
        embeddings_file = 'rule_embeddings.pkl'
        
        if os.path.exists(embeddings_file):
            try:
                with open(embeddings_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                st.warning(f"Could not load embeddings: {str(e)}. Computing new...")
                return _self.compute_embeddings()
        else:
            return _self.compute_embeddings()
                
    def compute_embeddings(self):
        """Compute embeddings for all rules"""
        if not self.model:
            return {}
            
        try:
            embeddings = {}
            with st.spinner("Computing embeddings for rules..."):
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
                    with open('rule_embeddings.pkl', 'wb') as f:
                        pickle.dump(embeddings, f)
                except Exception as e:
                    st.warning(f"Could not save embeddings: {str(e)}")
                    
            return embeddings
        except Exception as e:
            st.error(f"Error computing embeddings: {str(e)}")
            return {}
            
    def find_best_matches(self, query: str, language: str = 'fr', top_k: int = 3) -> List[RuleMatch]:
        """Find the best matching rules for a query"""
        if not self.model or not self.rule_embeddings:
            return []
            
        try:
            query_embedding = self.model.encode(query)
            matches = []
            
            for rule_id, rule_data in self.rule_embeddings.items():
                rule_embedding = rule_data[language]
                similarity = cosine_similarity([query_embedding], [rule_embedding])[0][0]
                
                matches.append(RuleMatch(
                    rule_id=rule_id,
                    score=similarity,
                    rule_data=rule_data['rule']
                ))
                
            matches.sort(key=lambda x: x.score, reverse=True)
            return matches[:top_k]
        except Exception as e:
            st.error(f"Error finding matches: {str(e)}")
            return []
        
    def extract_intent(self, query: str, language: str = 'fr') -> str:
        """Extract user intent from query"""
        query_lower = query.lower()
        
        intent_keywords = {
            'fr': {
                'scoring': ['point', 'score', 'calcul', 'comptage'],
                'cards': ['carte', 'valeur', 'atout', 'couleur'],
                'announcements': ['annonce', 'contrat', '90', '100', '110', '120', '130', '140', 'recommandations'],
                'coinche': ['coinche', 'surcoinche', 'multiplicateur'],
                'strategy': ['stratégie', 'conseil', 'astuce', 'tactique'],
                'basic': ['règle', 'base', 'comment', 'début', 'jeu'],
                'examples': ['exemple', 'cas', 'situation', 'pratique'],
                'capot': ['capot', 'tous', 'plis']
            },
            'en': {
                'scoring': ['point', 'score', 'calculate', 'counting'],
                'cards': ['card', 'value', 'trump', 'color'],
                'announcements': ['announcement', 'contract', '90', '100', '110', '120', '130', '140', 'recommendations'],
                'coinche': ['coinche', 'surcoinche', 'multiplier'],
                'strategy': ['strategy', 'advice', 'tip', 'tactic'],
                'basic': ['rule', 'basic', 'how', 'start', 'game'],
                'examples': ['example', 'case', 'situation', 'practical'],
                'capot': ['capot', 'all', 'tricks']
            }
        }
        
        keywords = intent_keywords.get(language, intent_keywords['fr'])
        
        for intent, words in keywords.items():
            if any(word in query_lower for word in words):
                return intent
                
        return 'general'
        
    def generate_contextual_response(self, matches: List[RuleMatch], intent: str, 
                                   language: str = 'fr', context: List[str] = None) -> str:
        """Generate a contextual response based on matches and intent"""
        if not matches or matches[0].score < 0.3:
            return self.get_fallback_response(intent, language, context)
            
        best_match = matches[0]
        rule = best_match.rule_data
        
        title = rule['title_fr'] if language == 'fr' else rule['title_en']
        content = rule['content_fr'] if language == 'fr' else rule['content_en']
        examples = rule.get('examples_fr', []) if language == 'fr' else rule.get('examples_en', [])
        
        response = f"**{title}**\n\n{content}"
        
        if examples and (intent == 'examples' or best_match.score > 0.7):
            example_header = "**Exemples:**" if language == 'fr' else "**Examples:**"
            response += f"\n\n{example_header}\n"
            for example in examples[:2]:
                response += f"• {example}\n"
                
        if best_match.score > 0.8 and len(matches) > 1:
            related_header = "**Voir aussi:**" if language == 'fr' else "**See also:**"
            response += f"\n\n{related_header}\n"
            for match in matches[1:2]:
                related_title = match.rule_data['title_fr'] if language == 'fr' else match.rule_data['title_en']
                response += f"• {related_title}\n"
                
        return response
        
    def get_fallback_response(self, intent: str, language: str = 'fr', context: List[str] = None) -> str:
        """Generate fallback response when no good match is found"""
        fallbacks = {
            'fr': {
                'scoring': "Je peux vous expliquer le système de scoring de la Belote! Voulez-vous savoir comment calculer les points des cartes, le système de contrats, ou les bonus?",
                'cards': "Les cartes ont des valeurs différentes selon qu'elles sont à l'atout ou non. Voulez-vous connaître les valeurs spécifiques?",
                'announcements': "Les annonces vont de 90 à 140 points. Chaque niveau a ses propres recommandations. Quel niveau vous intéresse?",
                'coinche': "Le système Coinche permet de multiplier les gains et les risques. Voulez-vous en savoir plus sur les multiplicateurs?",
                'strategy': "Je peux partager des stratégies et conseils pour améliorer votre jeu. Quel aspect vous intéresse le plus?",
                'basic': "Je peux expliquer les règles de base de la Belote Contrée. Par quoi voulez-vous commencer?",
                'general': "Je ne suis pas sûr de comprendre votre question. Pouvez-vous être plus spécifique? Je peux vous aider avec les règles, le scoring, les annonces, ou les stratégies."
            },
            'en': {
                'scoring': "I can explain the Belote scoring system! Would you like to know about card point calculation, contract system, or bonuses?",
                'cards': "Cards have different values depending on whether they're trump or not. Would you like to know the specific values?",
                'announcements': "Announcements range from 90 to 140 points. Each level has its own recommendations. Which level interests you?",
                'coinche': "The Coinche system allows multiplying gains and risks. Would you like to know more about multipliers?",
                'strategy': "I can share strategies and tips to improve your game. What aspect interests you most?",
                'basic': "I can explain the basic rules of Belote Contrée. Where would you like to start?",
                'general': "I'm not sure I understand your question. Could you be more specific? I can help with rules, scoring, announcements, or strategies."
            }
        }
        
        return fallbacks.get(language, fallbacks['fr']).get(intent, fallbacks[language]['general'])
        
    async def process_query(self, query: str, language: str = 'fr', context: List[str] = None) -> str:
        """Main method to process user queries"""
        intent = self.extract_intent(query, language)
        matches = self.find_best_matches(query, language)
        response = self.generate_contextual_response(matches, intent, language, context)
        return response

def init_session_state():
    """Initialize Streamlit session state"""
    if 'conversation' not in st.session_state:
        st.session_state.conversation = ConversationManager()
    if 'ai' not in st.session_state:
        st.session_state.ai = BeloteAI()
        st.session_state.ai.rule_embeddings = st.session_state.ai.initialize_embeddings()
    if 'language' not in st.session_state:
        st.session_state.language = 'fr'
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def get_quick_suggestions(language: str):
    """Get quick question suggestions based on language"""
    if language == 'fr':
        return [
            "Comment calculer les points?",
            "Valeurs des cartes à l'atout",
            "Que signifie coinche?",
            "Recommandations pour annonce 120",
            "Règles du capot"
        ]
    else:
        return [
            "How to calculate points?",
            "Trump card values",
            "What is coinche?",
            "Recommendations for 120 announcement",
            "Capot rules"
        ]

async def process_message(message: str):
    """Process a message and add response to chat"""
    st.session_state.messages.append({"role": "user", "content": message})
    st.session_state.conversation.add_message("user", message)
    
    context = st.session_state.conversation.get_context()
    response = await st.session_state.ai.process_query(message, st.session_state.language, context)
    
    st.session_state.conversation.add_message("bot", response)
    st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="Tunisian Belote Bot",
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
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.title("🎮 Belote Bot")
        
        current_lang = st.session_state.language
        if st.button(f"🌍 {'Français' if current_lang == 'en' else 'English'}"):
            st.session_state.language = 'en' if current_lang == 'fr' else 'fr'
            st.rerun()
        
        st.divider()
        
        suggestions_title = "Questions rapides:" if st.session_state.language == 'fr' else "Quick questions:"
        st.subheader(suggestions_title)
        
        suggestions = get_quick_suggestions(st.session_state.language)
        for i, suggestion in enumerate(suggestions):
            if st.button(suggestion, key=f"suggestion_{i}_{st.session_state.language}"):
                asyncio.run(process_message(suggestion))
                st.rerun()
        
        st.divider()
        
        if st.button("🗑️ " + ("Effacer l'historique" if st.session_state.language == 'fr' else "Clear History")):
            st.session_state.conversation.clear_history()
            st.session_state.messages = []
            st.success("Historique effacé!" if st.session_state.language == 'fr' else "History cleared!")
            st.rerun()
        
        if st.button("💾 " + ("Exporter" if st.session_state.language == 'fr' else "Export")):
            filename = f"belote_conversation_{st.session_state.conversation.get_timestamp()}.txt"
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
                except:
                    pass
        
        summary = st.session_state.conversation.get_conversation_summary(st.session_state.language)
        st.info(summary)
    
    if st.session_state.language == 'fr':
        st.title("🎮 Assistant Belote Tunisienne")
        st.markdown("""
        Bienvenue! Je peux vous aider avec:
        • Règles de base et gameplay
        • Calcul des scores et points
        • Système d'annonces (90-140)
        • Coinche et surcoinche
        • Stratégies et conseils
        • Valeurs des cartes
        """)
    else:
        st.title("🎮 Tunisian Belote Assistant")
        st.markdown("""
        Welcome! I can help you with:
        • Basic rules and gameplay
        • Score calculation and points
        • Announcement system (90-140)
        • Coinche and surcoinche
        • Strategies and tips
        • Card values
        """)
    
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    prompt_text = "Votre question..." if st.session_state.language == 'fr' else "Your question..."
    
    if prompt := st.chat_input(prompt_text):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Réflexion..." if st.session_state.language == 'fr' else "Thinking..."):
                try:
                    asyncio.run(process_message(prompt))
                    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                        st.markdown(st.session_state.messages[-1]["content"])
                        
                except Exception as e:
                    error_msg = f"Erreur: {str(e)}" if st.session_state.language == 'fr' else f"Error: {str(e)}"
                    st.error(error_msg)
    
    st.divider()
    if st.session_state.language == 'fr':
        st.markdown("*Développé avec ❤️ pour les amateurs de Belote Tunisienne*")
    else:
        st.markdown("*Developed with ❤️ for Tunisian Belote enthusiasts*")

if __name__ == "__main__":
    main()