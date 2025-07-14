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
            "basic_rules": {
                "title_fr": "Règles de Base de la Belote Contrée",
                "title_en": "Basic Rules of Belote Contrée", 
                "content_fr": "La Belote Contrée se joue à 4 joueurs en équipe de 2. Chaque joueur reçoit 8 cartes. L'objectif est de réaliser le contrat annoncé en marquant suffisamment de points avec les plis remportés.",
                "content_en": "Belote Contrée is played with 4 players in teams of 2. Each player receives 8 cards. The objective is to fulfill the announced contract by scoring enough points with the tricks won.",
                "keywords_fr": ["règles", "base", "équipe", "cartes", "plis", "contrat"],
                "keywords_en": ["rules", "basic", "team", "cards", "tricks", "contract"],
                "examples_fr": [
                    "Équipe Nord-Sud vs Équipe Est-Ouest",
                    "Distribution: 3+2+3 cartes par tour"
                ],
                "examples_en": [
                    "North-South Team vs East-West Team", 
                    "Deal: 3+2+3 cards per round"
                ]
            },
            "card_values": {
                "title_fr": "Valeurs des Cartes",
                "title_en": "Card Values",
                "content_fr": "À l'atout: Valet(20), 9(14), As(11), 10(10), Roi(4), Dame(3), 8(0), 7(0). Hors atout: As(11), 10(10), Roi(4), Dame(3), Valet(2), 9(0), 8(0), 7(0). Total: 162 points par donne.",
                "content_en": "Trump suit: Jack(20), 9(14), Ace(11), 10(10), King(4), Queen(3), 8(0), 7(0). Non-trump: Ace(11), 10(10), King(4), Queen(3), Jack(2), 9(0), 8(0), 7(0). Total: 162 points per deal.",
                "keywords_fr": ["valeurs", "cartes", "atout", "points", "valet", "as"],
                "keywords_en": ["values", "cards", "trump", "points", "jack", "ace"],
                "examples_fr": [
                    "Valet d'atout = 20 points (carte la plus forte)",
                    "9 d'atout = 14 points (2ème plus forte)"
                ],
                "examples_en": [
                    "Trump Jack = 20 points (strongest card)",
                    "Trump 9 = 14 points (2nd strongest)"
                ]
            },
            "announcements": {
                "title_fr": "Système d'Annonces (90-140)",
                "title_en": "Announcement System (90-140)",
                "content_fr": "Les contrats vont de 90 à 140 points. 90: jeu faible mais régulier. 100-110: jeu moyen avec atouts. 120-130: beau jeu avec honneurs. 140: jeu exceptionnel ou capot possible.",
                "content_en": "Contracts range from 90 to 140 points. 90: weak but regular game. 100-110: average game with trumps. 120-130: good game with honors. 140: exceptional game or possible capot.",
                "keywords_fr": ["annonces", "contrat", "90", "100", "110", "120", "130", "140"],
                "keywords_en": ["announcements", "contract", "90", "100", "110", "120", "130", "140"],
                "examples_fr": [
                    "Annonce 120 avec As-Roi d'atout et belle répartition",
                    "Annonce 90 pour gêner l'adversaire"
                ],
                "examples_en": [
                    "120 announcement with Ace-King of trump and good distribution",
                    "90 announcement to disturb opponents"
                ]
            },
            "coinche": {
                "title_fr": "Coinche et Surcoinche",
                "title_en": "Coinche and Surcoinche",
                "content_fr": "Coinche: doubler la mise de l'adversaire (multiplicateur x2). Surcoinche: redoubler après une coinche (multiplicateur x4). Attention: les risques sont aussi multipliés!",
                "content_en": "Coinche: double the opponent's bet (x2 multiplier). Surcoinche: redouble after a coinche (x4 multiplier). Warning: risks are also multiplied!",
                "keywords_fr": ["coinche", "surcoinche", "doubler", "multiplicateur"],
                "keywords_en": ["coinche", "surcoinche", "double", "multiplier"],
                "examples_fr": [
                    "Adversaire annonce 100, vous cochez → 200 points",
                    "Vous surcoinchez → 400 points si ils chutent"
                ],
                "examples_en": [
                    "Opponent announces 100, you coinche → 200 points",
                    "You surcoinche → 400 points if they fail"
                ]
            },
            "scoring": {
                "title_fr": "Calcul des Scores",
                "title_en": "Score Calculation", 
                "content_fr": "Score = points des plis + bonus/malus. Contrat réussi: points annoncés + points réels. Contrat chuté: 0 point + adversaires marquent 162 + contrat. Capot: +250 points.",
                "content_en": "Score = trick points + bonus/penalty. Contract made: announced points + actual points. Contract failed: 0 points + opponents score 162 + contract. Capot: +250 points.",
                "keywords_fr": ["score", "calcul", "points", "contrat", "capot", "bonus"],
                "keywords_en": ["score", "calculation", "points", "contract", "capot", "bonus"],
                "examples_fr": [
                    "Contrat 120 réussi avec 135 points → 120+135 = 255 points",
                    "Contrat 100 chuté → 0 point, adversaires: 162+100 = 262"
                ],
                "examples_en": [
                    "120 contract made with 135 points → 120+135 = 255 points", 
                    "100 contract failed → 0 points, opponents: 162+100 = 262"
                ]
            },
            "strategy": {
                "title_fr": "Stratégies et Conseils",
                "title_en": "Strategies and Tips",
                "content_fr": "Comptez les atouts sortis. Jouez vos gros atouts en premier. Défaussez dans la couleur faible de l'adversaire. Communiquez avec votre partenaire par le jeu.",
                "content_en": "Count trumps played. Play your high trumps first. Discard in opponent's weak suit. Communicate with partner through card play.",
                "keywords_fr": ["stratégie", "conseils", "atouts", "communication", "défausse"],
                "keywords_en": ["strategy", "tips", "trumps", "communication", "discard"],
                "examples_fr": [
                    "Entamer par l'As pour voir la réaction du partenaire",
                    "Couper avec petit atout pour économiser les gros"
                ],
                "examples_en": [
                    "Lead with Ace to see partner's reaction",
                    "Trump with small trump to save the big ones"
                ]
            }
        }
    
    def get_all_rules(self):
        return self.rules
    
    def get_rule(self, rule_id: str):
        return self.rules.get(rule_id)

class ConversationManager:
    """Conversation management and context handling"""
    
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.context_window = 5  # Remember last 5 exchanges
        
    def add_message(self, sender: str, content: str):
        """Add a message to conversation history"""
        message = {
            'sender': sender,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        self.messages.append(message)
        
        # Keep only recent messages to manage memory
        if len(self.messages) > self.context_window * 2:  # user + bot messages
            self.messages = self.messages[-self.context_window * 2:]
            
    def get_context(self) -> List[str]:
        """Get recent conversation context"""
        recent_messages = self.messages[-self.context_window:]
        return [msg['content'] for msg in recent_messages if msg['sender'] == 'user']
        
    def get_last_user_messages(self, count: int = 3) -> List[str]:
        """Get last N user messages"""
        user_messages = [msg['content'] for msg in self.messages if msg['sender'] == 'user']
        return user_messages[-count:] if user_messages else []
        
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

class BeloteAI:
    """Advanced AI engine for Tunisian Belote rules using sentence transformers"""
    
    def __init__(self):
        if DEPENDENCIES_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                st.error(f"Error loading SentenceTransformer model: {str(e)}")
                self.model = None
        else:
            self.model = None
            
        self.rules_db = BeloteRulesDatabase()
        self.rule_embeddings = {}
        self.context_window = 3  # Remember last 3 exchanges
        
        if self.model:
            self.initialize_embeddings()
        
    def initialize_embeddings(self):
        """Initialize or load pre-computed embeddings for all rules"""
        embeddings_file = 'rule_embeddings.pkl'
        
        if os.path.exists(embeddings_file):
            try:
                with open(embeddings_file, 'rb') as f:
                    self.rule_embeddings = pickle.load(f)
            except Exception as e:
                st.warning(f"Could not load embeddings file: {str(e)}. Computing new embeddings...")
                self.compute_embeddings()
        else:
            self.compute_embeddings()
                
    def compute_embeddings(self):
        """Compute embeddings for all rules"""
        if not self.model:
            st.error("Cannot compute embeddings: model not available")
            return
            
        try:
            with st.spinner("Computing embeddings for rules... This may take a moment."):
                for rule_id, rule in self.rules_db.get_all_rules().items():
                    # Combine title, content, and keywords for embedding
                    text_fr = f"{rule['title_fr']} {rule['content_fr']} {' '.join(rule['keywords_fr'])}"
                    text_en = f"{rule['title_en']} {rule['content_en']} {' '.join(rule['keywords_en'])}"
                    
                    embedding_fr = self.model.encode(text_fr)
                    embedding_en = self.model.encode(text_en)
                    
                    self.rule_embeddings[rule_id] = {
                        'fr': embedding_fr,
                        'en': embedding_en,
                        'rule': rule
                    }
                
                # Save embeddings
                try:
                    with open('rule_embeddings.pkl', 'wb') as f:
                        pickle.dump(self.rule_embeddings, f)
                except Exception as e:
                    st.warning(f"Could not save embeddings: {str(e)}")
                    
        except Exception as e:
            st.error(f"Error computing embeddings: {str(e)}")
            
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
                
            # Sort by similarity score and return top_k
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
                'announcements': ['annonce', 'contrat', '90', '100', '110', '120', '130', '140'],
                'coinche': ['coinche', 'surcoinche', 'multiplicateur'],
                'strategy': ['stratégie', 'conseil', 'astuce', 'tactique'],
                'basic': ['règle', 'base', 'comment', 'début', 'jeu'],
                'examples': ['exemple', 'cas', 'situation', 'pratique']
            },
            'en': {
                'scoring': ['point', 'score', 'calculate', 'counting'],
                'cards': ['card', 'value', 'trump', 'color'],
                'announcements': ['announcement', 'contract', '90', '100', '110', '120', '130', '140'],
                'coinche': ['coinche', 'surcoinche', 'multiplier'],
                'strategy': ['strategy', 'advice', 'tip', 'tactic'],
                'basic': ['rule', 'basic', 'how', 'start', 'game'],
                'examples': ['example', 'case', 'situation', 'practical']
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
        
        # Select content based on language
        title = rule['title_fr'] if language == 'fr' else rule['title_en']
        content = rule['content_fr'] if language == 'fr' else rule['content_en']
        examples = rule.get('examples_fr', []) if language == 'fr' else rule.get('examples_en', [])
        
        response = f"**{title}**\n\n{content}"
        
        # Add examples if available and relevant
        if examples and (intent == 'examples' or best_match.score > 0.7):
            example_header = "**Exemples:**" if language == 'fr' else "**Examples:**"
            response += f"\n\n{example_header}\n"
            for example in examples[:2]:  # Limit to 2 examples
                response += f"• {example}\n"
                
        # Add related suggestions if confidence is high
        if best_match.score > 0.8 and len(matches) > 1:
            related_header = "**Voir aussi:**" if language == 'fr' else "**See also:**"
            response += f"\n\n{related_header}\n"
            for match in matches[1:2]:  # Add one related topic
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
        # Extract intent
        intent = self.extract_intent(query, language)
        
        # Find best matching rules
        matches = self.find_best_matches(query, language)
        
        # Generate contextual response
        response = self.generate_contextual_response(matches, intent, language, context)
        
        return response

def init_session_state():
    """Initialize Streamlit session state"""
    if 'conversation' not in st.session_state:
        st.session_state.conversation = ConversationManager()
    if 'ai' not in st.session_state:
        st.session_state.ai = BeloteAI()
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

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Tunisian Belote Bot",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Custom CSS
    st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        margin-bottom: 5px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .bot-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("🎮 Belote Bot")
        
        # Language toggle
        current_lang = st.session_state.language
        if st.button(f"🌍 {'Français' if current_lang == 'en' else 'English'}"):
            st.session_state.language = 'en' if current_lang == 'fr' else 'fr'
            st.rerun()
        
        st.divider()
        
        # Quick suggestions
        suggestions_title = "Questions rapides:" if st.session_state.language == 'fr' else "Quick questions:"
        st.subheader(suggestions_title)
        
        suggestions = get_quick_suggestions(st.session_state.language)
        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggestion_{suggestion}"):
                # Add suggestion to chat
                st.session_state.messages.append({"role": "user", "content": suggestion})
                st.session_state.conversation.add_message("user", suggestion)
                st.rerun()
        
        st.divider()
        
        # Controls
        if st.button("🗑️ " + ("Effacer l'historique" if st.session_state.language == 'fr' else "Clear History")):
            st.session_state.conversation.clear_history()
            st.session_state.messages = []
            st.success("Historique effacé!" if st.session_state.language == 'fr' else "History cleared!")
            st.rerun()
        
        if st.button("💾 " + ("Exporter" if st.session_state.language == 'fr' else "Export")):
            filename = f"belote_conversation_{st.session_state.conversation.get_timestamp()}.txt"
            if st.session_state.conversation.export_to_file(filename, st.session_state.language):
                st.success(f"Exporté vers {filename}" if st.session_state.language == 'fr' else f"Exported to {filename}")
                
                # Offer download
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
        
        # Conversation summary
        summary = st.session_state.conversation.get_conversation_summary(st.session_state.language)
        st.info(summary)
    
    # Main content
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
    
    # Chat interface
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    prompt_text = "Votre question..." if st.session_state.language == 'fr' else "Your question..."
    
    if prompt := st.chat_input(prompt_text):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.conversation.add_message("user", prompt)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate bot response
        with st.chat_message("assistant"):
            with st.spinner("Réflexion..." if st.session_state.language == 'fr' else "Thinking..."):
                try:
                    # Get context
                    context = st.session_state.conversation.get_context()
                    
                    # Process query
                    response = asyncio.run(
                        st.session_state.ai.process_query(prompt, st.session_state.language, context)
                    )
                    
                    # Add to conversation
                    st.session_state.conversation.add_message("bot", response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    st.markdown(response)
                    
                except Exception as e:
                    error_msg = f"Erreur: {str(e)}" if st.session_state.language == 'fr' else f"Error: {str(e)}"
                    st.error(error_msg)
    
    # Footer
    st.divider()
    if st.session_state.language == 'fr':
        st.markdown("*Développé avec ❤️ pour les amateurs de Belote Tunisienne*")
    else:
        st.markdown("*Developed with ❤️ for Tunisian Belote enthusiasts*")

if __name__ == "__main__":
    main()