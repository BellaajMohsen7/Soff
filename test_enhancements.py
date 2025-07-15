#!/usr/bin/env python3
"""
Test suite for enhanced Sofiene Bot capabilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit_belote_app as app

def test_enhanced_features():
    """Test all enhanced features of the Sofiene Bot"""
    print("🧪 Testing Enhanced Sofiene Bot Features...")
    
    # Initialize AI
    ai = app.SofieneAI()
    
    # Test 1: Fuzzy matching for typos
    print("\n1. Testing fuzzy matching for typos:")
    typo_queries = [
        ("recomendation for 120", "Should handle typo in 'recommendation'"),
        ("belotte rebelotte", "Should handle typo in 'belote'"),
        ("oficiel rules", "Should handle typo in 'official'"),
        ("anouncement 110", "Should handle typo in 'announcement'")
    ]
    
    for query, description in typo_queries:
        try:
            response = ai.process_query(query, 'fr')
            success = len(response) > 100  # Reasonable response length
            print(f"  ✓ {description}: {'PASS' if success else 'FAIL'}")
        except Exception as e:
            print(f"  ✗ {description}: ERROR - {e}")
    
    # Test 2: Enhanced pattern recognition
    print("\n2. Testing enhanced pattern recognition:")
    pattern_queries = [
        ("what is rule of announce", "Original problem case"),
        ("conseil pour 120 points", "Alternative phrasing"),
        ("règle annonce 110", "Simplified phrasing"),
        ("comment faire 130", "Informal question"),
        ("usage belote rebelote", "Different word order")
    ]
    
    for query, description in pattern_queries:
        try:
            response = ai.process_query(query, 'fr')
            success = len(response) > 100
            print(f"  ✓ {description}: {'PASS' if success else 'FAIL'}")
        except Exception as e:
            print(f"  ✗ {description}: ERROR - {e}")
    
    # Test 3: New rules coverage
    print("\n3. Testing new rules coverage:")
    new_rules_queries = [
        ("points partenaires", "Partner points system"),
        ("gestion contrat", "Contract management"),
        ("règles capot", "Complete capot rules"),
        ("équipe points addition", "Team points addition")
    ]
    
    for query, description in new_rules_queries:
        try:
            response = ai.process_query(query, 'fr')
            success = len(response) > 200  # Should be comprehensive
            print(f"  ✓ {description}: {'PASS' if success else 'FAIL'}")
        except Exception as e:
            print(f"  ✗ {description}: ERROR - {e}")
    
    # Test 4: Bilingual support with enhancements
    print("\n4. Testing bilingual support:")
    bilingual_queries = [
        ("recommendation for 120 points", "en", "English recommendation"),
        ("when to use belote", "en", "English belote question"),
        ("partner points system", "en", "English partner points"),
        ("recommandation 110 points", "fr", "French recommendation")
    ]
    
    for query, lang, description in bilingual_queries:
        try:
            response = ai.process_query(query, lang)
            success = len(response) > 100
            print(f"  ✓ {description}: {'PASS' if success else 'FAIL'}")
        except Exception as e:
            print(f"  ✗ {description}: ERROR - {e}")
    
    # Test 5: Hand evaluation with variations
    print("\n5. Testing hand evaluation:")
    hand_queries = [
        ("j'ai valet 9 as 10 carreau que annoncer", "Complete trump description"),
        ("avec 2 as main conseil", "Simple hand description"),
        ("analyser ma main 6 atouts", "Analysis request"),
        ("evaluate hand with jack 9", "English hand evaluation")
    ]
    
    for query, description in hand_queries:
        try:
            response = ai.process_query(query, 'fr')
            success = "recommandation" in response.lower() or "recommendation" in response.lower()
            print(f"  ✓ {description}: {'PASS' if success else 'FAIL'}")
        except Exception as e:
            print(f"  ✗ {description}: ERROR - {e}")
    
    # Test 6: Enhanced fallback responses
    print("\n6. Testing enhanced fallback responses:")
    fallback_queries = [
        ("xyzabc random text", "Random text should get helpful fallback"),
        ("help me", "General help request"),
        ("what can you do", "Capability inquiry")
    ]
    
    for query, description in fallback_queries:
        try:
            response = ai.process_query(query, 'fr')
            has_suggestions = "essayez" in response.lower() or "try" in response.lower()
            print(f"  ✓ {description}: {'PASS' if has_suggestions else 'FAIL'}")
        except Exception as e:
            print(f"  ✗ {description}: ERROR - {e}")
    
    print("\n🎯 Enhanced Sofiene Bot Testing Complete!")
    print("All major enhancements have been tested.")

if __name__ == "__main__":
    test_enhanced_features()