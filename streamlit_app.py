
import streamlit as st
st.set_page_config(layout="wide")
from torch.nn.functional import softmax
import pandas as pd
import screed
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

       
def change_aa(peptide, index, aa):
    return peptide[:index] + aa + peptide[index+1:]


def mutant_peptides(peptide):
    alphabet = ['G', 'I', 'V', 'F', 'Y', 'W', 'A', 'L', 'M', 'E', 'Q', 'R', 'K', 'P', 'N', 'D', 'H', 'S', 'T', 'C']
    mutants = []
    for aa in alphabet:
        for i in range(len(peptide)):        
            if aa != peptide[i]:
                mutants.append( change_aa(peptide, i, aa) )
                
    return mutants 
  
def clear_sequence(sequence):
    alphabet = ['G', 'I', 'V', 'F', 'Y', 'W', 'A', 'L', 'M', 'E', 'Q', 'R', 'K', 'P', 'N', 'D', 'H', 'S', 'T', 'C']
    sequence = sequence.upper()
    clear_seq = ""
    for i in sequence:
        if i in alphabet:
            clear_seq += i
    return clear_seq
    
#########################################################################################################################################


model_path = "./esm2_t6_8M_UR50D-finetuned-antioxidantes"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

labels_model = ['Não antioxidante', 'Antioxidante'] 
 
st.markdown("""
    <style>
        button {
            height: auto;
            width: 100% !important;
            
        }
        p {
            text-align: justify
        }
    </style>
""", unsafe_allow_html=True) 
 
st.header('Predição e otimização de peptídeos antioxidantes')
st.markdown("""<p>Na aba "Classificador", é possível avaliar se peptídeos tem potencial ação antioxidante. A ferramenta
 pode receber uma ou mais sequências peptídicas em formato FASTA. </p>
 <p>Na aba "Peptídeos mutantes", a ferramenta pode ser usada para prever a atividade antioxidante de um peptídeo e seus análogos com 
 sucessivas substituições de aminoácidos em cada posição. Esse recurso ajuda o usuário a selecionar os peptídeos mutantes, em relação ao peptídeo original, que podem
 ter uma probabilidade mais alta de apresentar atividade antioxidante.</p>
 """, unsafe_allow_html=True) 


# Criando guias
guias = st.tabs(["Classificador", "Peptídeos mutantes"])

# Conteúdo das guias
with guias[0]:
    if "sequences" not in st.session_state:
        st.session_state["sequences"] = ""

    sequences_area = st.text_area("Cole sua sequência em formato FASTA ou use o exemplo", value = st.session_state["sequences"], height = 300)
        
    query_sequences = []
    query_labels = []

    br = st.button("Executar", type="primary")
    ex = st.button("Use um exemplo")
    cl = st.button("Limpar")

    if br:
        progress_text = "Processando ... "
        print(progress_text)
        
        start_time = time.time()
        temp = open("temp.fas", "w")
        temp.write(sequences_area.strip())
        temp.close()
        
        query_labels = []
        query_sequences =[]

        for record in screed.open("temp.fas"):
            name = record.name
            sequence = record.sequence
            
            query_labels.append(name)
            query_sequences.append(sequence)
            
        n_queries = len(query_sequences)
         
        
        my_bar1 = st.progress(0, text="")
        counter = 0
        query_name = []        
        predicted_class = []
        probabilities_nao_antioxidante = []
        probabilities_antioxidante = []
        for s in range(len(query_sequences)):
            counter += 1
            my_bar1.progress(round( (counter/len(query_sequences))*100 ), text=progress_text + str(counter) + " de " + str(len(query_sequences)))
                        
            query_seq = clear_sequence(query_sequences[s])  # limpar a sequência
            tokenized_query_seq = tokenizer([query_seq], padding=True, truncation=True, return_tensors="pt") # tokenizar a sequência
            
            # Obter as previsões do modelo
            outputs = model(**tokenized_query_seq)
            logits = outputs.logits

            # Calcular as probabilidades 
            probabilities = softmax(logits, dim=1)
            
            # Predição
            predictions = probabilities.argmax(dim=1)        
                     
            # Adicionando resultados                
            query_name.append(query_labels[s])
            
            predicted_class.append(labels_model[int(predictions[0].item())])
            
            probabilities_nao_antioxidante.append(round (float(probabilities[0][0].item()), 3 ) )
            
            probabilities_antioxidante.append( round( float(probabilities[0][1].item()), 3 ) )
        
        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        st.write(f"Tempo de execução: {int(hours)} horas, {int(minutes)} minutos, {int(seconds)} segundos")
        
        d = {'Nome da sequência de consulta': query_name, 'Classe predita': predicted_class, 'Probabilidade: não antioxidante': probabilities_nao_antioxidante,
            'Probabilidade: antioxidante': probabilities_antioxidante }        
        
        df = pd.DataFrame(data=d,index=None)
                
        st.table(df)
        

     
    example = """>nao_antioxidante_1
VQRELLDAAIAAK
>nao_antioxidante_2
IRSIKKNWEFQAI
>nao_antioxidante_3
TDQLQALGIKVFGPTKAGAELEASKSW
>nao_antioxidante_4
FGTPVDSQQAPE
>nao_antioxidante_5
PRQLTHDAGLTVDDIAKRLMDYG
>antioxidante_1
VLPVPQKKVLPVPQK
>antioxidante_2
SALLRSIPA
>antioxidante_3
PSHDAHPE
>antioxidante_4
PHHADS
>antioxidante_5
DRVYIHPFHL
    """


    if ex:
        st.session_state["sequences"] = example
        st.rerun()  


    if cl:
        st.session_state["sequences"] = ""
        st.rerun() 
        
with guias[1]:
    if "sequence_mutant" not in st.session_state:
        st.session_state["sequence_mutant"] = ""

    sequences_area_m = st.text_area("Cole sua sequência em formato FASTA ou use o exemplo", value = st.session_state["sequence_mutant"], height = 100)
        
    query_sequences = []
    query_labels = []

    br_m = st.button("Criar peptídeos mutantes e realizar a classificação", type="primary")
    ex_m = st.button("Usar um exemplo")
    cl_m = st.button("Limpar o formulário")

    if br_m:
        progress_text = "Processando ... "
        print(progress_text)
        
        start_time = time.time()
        temp_m = open("temp_m.fas", "w")
        temp_m.write(sequences_area_m.strip())
        temp_m.close()
        
        query_labels = []
        query_sequences =[]

        for record in screed.open("temp_m.fas"):
            name = record.name
            sequence = record.sequence
            break
          
        query_sequences.append(sequence)
        query_sequences = query_sequences + mutant_peptides(sequence)    
        
        n_queries = len(query_sequences)        
        
        query_labels.append(name)
        for i in range(n_queries):
            query_labels.append(name+"_mutant_"+str(i+1))
        
        my_bar1_m = st.progress(0, text="")
        counter = 0
        query_name = []
        query_mutant = []
        predicted_class = []
        probabilities_nao_antioxidante = []
        probabilities_antioxidante = []
        for s in range(len(query_sequences)):
            counter += 1
            my_bar1_m.progress(round( (counter/len(query_sequences))*100 ), text=progress_text + str(counter) + " of " + str(len(query_sequences)))
            
            query_seq = clear_sequence(query_sequences[s])  # limpar a sequência
            tokenized_query_seq = tokenizer([query_seq], padding=True, truncation=True, return_tensors="pt") # tokenizar a sequência
            
            # Obter as previsões do modelo
            outputs = model(**tokenized_query_seq)
            logits = outputs.logits

            # Calcular as probabilidades 
            probabilities = softmax(logits, dim=1)
            
            # Predição
            predictions = probabilities.argmax(dim=1)        
                     
            # Adicionando resultados                
            query_name.append(query_labels[s])
            
            query_mutant.append(query_sequences[s])
            
            predicted_class.append(labels_model[int(predictions[0].item())])
            
            probabilities_nao_antioxidante.append( round( float(probabilities[0][0].item()), 3 ) )
            
            probabilities_antioxidante.append( round( float(probabilities[0][1].item()), 3 ) )
            
            
        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        st.write(f"Tempo de execução: {int(hours)} horas, {int(minutes)} minutos, {int(seconds)} segundos")
        
                
        d = {'Nome da sequência de consulta': query_name, 'Sequência peptídica': query_mutant, 'Classe predita': predicted_class,
            'Probabilidade: não antioxidante': probabilities_nao_antioxidante, 'Probabilidade: antioxidante': probabilities_antioxidante }
        df = pd.DataFrame(data=d,index=None)
        
        
        with st.expander(":blue[**Melhores resultados**]"):
            df_pep_selvagem = df.iloc[0]
            st.write("Dados do peptídeo original (selvagem):")
            st.write("Nome: ", df_pep_selvagem["Nome da sequência de consulta"]) 
            st.write("Sequência peptídica: ", df_pep_selvagem["Sequência peptídica"])
            st.write("Classe predita: ", df_pep_selvagem["Classe predita"])
            st.write("Probabilidade não antioxidante: ", str(df_pep_selvagem["Probabilidade: não antioxidante"]))
            st.write("Probabilidade antioxidante: ", str(df_pep_selvagem["Probabilidade: antioxidante"]))
            
            st.write("\n\n")
            st.write("Melhores sequências mutantes encontradas:")
            st.write( df.loc[(df['Probabilidade: antioxidante'] > df_pep_selvagem["Probabilidade: antioxidante"]) & (df['Classe predita'] == "Antioxidante")])
            
            
        
        with st.expander(":blue[**Resultados da varredurra completa**]"):
            st.table(df)
        

     
    example_m = """>peptide
RWRWRWF
    """


    if ex_m:
        st.session_state["sequence_mutant"] = example_m
        st.rerun()  


    if cl_m:
        st.session_state["sequence_mutant"] = ""
        st.rerun()










 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 
    




    
