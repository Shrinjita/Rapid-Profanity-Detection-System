# app.py
import streamlit as st
import pandas as pd
from detector import ProfanityDetector

def main():
    st.set_page_config(page_title="Profanity Detector", layout="wide")
    st.title("Multilingual Profanity Detection Demo")
    
    # Initialize detector
    detector = ProfanityDetector()
    
    # UI Layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Input")
        input_text = st.text_area("Enter text to analyze:", height=150)
        threshold = st.slider("Detection Threshold", 0.0, 5.0, 2.5, 0.1)
        
        # File upload
        uploaded_file = st.file_uploader("Or upload CSV file:", type=["csv"])
        
        # Quick test cases
        test_cases = st.selectbox("Quick test cases:", [
            "Select a test case",
            "Clean example: Hello world",
            "Explicit: This is shit",
            "Obfuscated: sh1t",
            "Punctuation: s.h.i.t",
            "Repeated: fuuuuck",
            "Substring: classroom",
            "Hindi: kutte ki aulad",
            "Hindi obfuscated: kutt3 k1 @ul@d"
        ])
        
        if test_cases != "Select a test case":
            input_text = test_cases.split(": ")[1]
    
    with col2:
        st.header("Results")
        
        if st.button("Analyze") or uploaded_file is not None:
            if uploaded_file is not None:
                # Batch processing
                df = pd.read_csv(uploaded_file)
                results = []
                for _, row in df.iterrows():
                    result = detector.detect(row['text'], threshold)
                    results.append({
                        'text': row['text'],
                        'flagged': result['flagged'],
                        'score': result['score']
                    })
                
                results_df = pd.DataFrame(results)
                flagged_count = results_df['flagged'].sum()
                
                st.subheader(f"Batch Results: {flagged_count}/{len(results_df)} flagged")
                st.dataframe(results_df)
                
                # Show some examples
                if flagged_count > 0:
                    st.subheader("Flagged Examples")
                    for i, row in results_df[results_df['flagged']].head(3).iterrows():
                        st.write(f"**Text:** {row['text']}")
                        st.write(f"**Score:** {row['score']}")
                        st.write("---")
            else:
                # Single text processing
                if input_text:
                    result = detector.detect(input_text, threshold)
                    
                    # Display result
                    if result['flagged']:
                        st.error(f"FLAGGED (Score: {result['score']:.2f})")
                    else:
                        st.success(f"CLEAN (Score: {result['score']:.2f})")
                    
                    # Show explanation
                    st.subheader("Explanation")
                    if result['matches']:
                        for match in result['matches']:
                            st.write(f"- **{match['type']}**: {match['matched']} → {match['original']} (weight: {match['weight']})")
                    else:
                        st.write("No profanity detected")
                    
                    # Show normalized text
                    st.subheader("Normalized Text")
                    st.code(result['normalized_text'])
                else:
                    st.warning("Please enter some text to analyze")

if __name__ == "__main__":
    main()