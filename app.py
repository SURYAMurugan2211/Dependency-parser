import streamlit as st
import stanza
import pandas as pd

# Download and load Stanza model
stanza.download('en')  # Ensure English model is downloaded
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')

# Streamlit UI
st.set_page_config(page_title="Dependency Parsing App", layout="wide")

st.title("Dependency Parsing Web App 📝")
st.write("Analyze the dependency structure of a sentence using **Stanza**.")

# User Input
sentence = st.text_area("Enter a sentence:", "The quick brown fox jumps over the lazy dog.")

if st.button("Parse Sentence"):
    if sentence.strip():
        doc = nlp(sentence)
        
        # Extract dependencies
        dependencies = []
        for sent in doc.sentences:
            for word in sent.words:
                dependencies.append({
                    "Token": word.text,
                    "Head": sent.words[word.head - 1].text if word.head > 0 else "ROOT",
                    "Relation": word.deprel
                })
        
        # Convert to DataFrame for better display
        df = pd.DataFrame(dependencies)
        st.write("### Dependency Parsing Result")
        st.dataframe(df)  # Display as a table
        
        # Dependency Tree Representation
        st.write("### Parsed Sentence Tree:")
        for dep in dependencies:
            st.write(f"**{dep['Token']}** → {dep['Head']} ({dep['Relation']})")
    else:
        st.warning("Please enter a valid sentence.")

# Footer
st.markdown("---")
st.markdown("Developed by **Surya** | NLP Project 🚀")
