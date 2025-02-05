import streamlit as st
import stanza
import pandas as pd

# Download and load Stanza model
stanza.download('en')  # Ensure English model is downloaded
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')

# Mapping of dependency relations to full descriptions
dependency_map = {
    "nsubj": "Nominal Subject",
    "amod": "Adjectival Modifier",
    "obl": "Oblique Nominal",
    "root": "Root of the Sentence",
    "case": "Case-Marking",
    "det": "Determiner",
    "punct": "Punctuation",
    "obj": "Object",
    "advmod": "Adverbial Modifier",
    "nmod": "Nominal Modifier",
    "cc": "Coordinating Conjunction",
    "conj": "Conjunct",
    "ccomp": "Clausal Complement",
    "xcomp": "Open Clausal Complement",
    "advcl": "Adverbial Clause Modifier",
    "mark": "Marker",
    "appos": "Appositional Modifier",
    "acl": "Clausal Modifier of Noun",
    "cop": "Copula",
    "expl": "Expletive",
    "aux": "Auxiliary",
    "csubj": "Clausal Subject",
    "dep": "Unspecified Dependency",
}

st.set_page_config(page_title="Dependency Parsing App", layout="wide")

st.title("Dependency Parsing Web App 📝")
st.write("Analyze the dependency structure of a sentence using **Stanza**.")

sentence = st.text_area("Enter a sentence:", "The quick brown fox jumps over the lazy dog.")

if st.button("Parse Sentence"):
    if sentence.strip():
        doc = nlp(sentence)
        
        dependencies = []
        for sent in doc.sentences:
            for word in sent.words:
                dependencies.append({
                    "Token": word.text,
                    "Head": sent.words[word.head - 1].text if word.head > 0 else "ROOT",
                    "Relation": dependency_map.get(word.deprel, word.deprel)  # Map to full name
                })
        
        df = pd.DataFrame(dependencies)
        st.write("### Dependency Parsing Result")
        st.dataframe(df)
        
        st.write("### Parsed Sentence Tree:")
        for dep in dependencies:
            st.write(f"**{dep['Token']}** → {dep['Head']} ({dep['Relation']})")
    else:
        st.warning("Please enter a valid sentence.")

st.markdown("---")
st.markdown("Developed by **Surya** | NLP Project 🚀")
