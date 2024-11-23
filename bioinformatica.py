import streamlit as st
from Bio import SeqIO  # type: ignore
import pandas as pd
import io
import plotly.express as px  # type: ignore
import tempfile
import os
from Bio.PDB import PDBParser
import numpy as np

# Título de la app
st.title("¡Sube tu archivo FASTA o PDB aquí!")
st.write("Por favor, sube tu archivo FASTA o PDB para comenzar el análisis.")

# File uploader: El usuario sube su archivo FASTA aquí
uploaded_fasta = st.file_uploader("Elige tu archivo FASTA", type=["fasta", "fa"])

def count_amino_acids(sequence):
    aa_count = {aa: sequence.count(aa) for aa in set(sequence)}
    return aa_count

if uploaded_fasta is not None:
    # Convertir el archivo subido a un flujo de texto y analizar el archivo FASTA
    try:
        with io.StringIO(uploaded_fasta.getvalue().decode("utf-8")) as stringio:
            record = SeqIO.read(stringio, "fasta")
    except Exception as e:
        st.error(f"Error al leer el archivo FASTA: {e}")
        record = None

    if record:
        st.subheader(f"ID de la secuencia: {record.id}")
        st.text_area("Secuencia", str(record.seq), height=200)

        # Selección de métodos de visualización
        st.subheader("Selecciona método(s) de visualización")
        show_3d_and_factor_b = st.checkbox("Modelo 3D y Factor B")
        show_table = st.checkbox("Tabla de secuencia")
        show_aa_distribution = st.checkbox("Distribución de Aminoácidos")
        show_sequence_alignment = st.checkbox("Alineamiento de Secuencias")
        show_secondary_structure = st.checkbox("Estructura Secundaria de la Proteína")

        if show_table:
            aa_count = count_amino_acids(str(record.seq))
            aa_df = pd.DataFrame(list(aa_count.items()), columns=["Aminoácido", "Repeticiones"])
            aa_df = aa_df.sort_values(by="Repeticiones", ascending=False)
            st.write("Tabla de Aminoácidos y sus Repeticiones")
            st.write(aa_df)

        if show_aa_distribution:
            aa_count = count_amino_acids(str(record.seq))
            aa_labels = list(aa_count.keys())
            aa_values = list(aa_count.values())
            total_aa = sum(aa_values)
            aa_percentages = [round((value / total_aa) * 100, 2) for value in aa_values]

            aa_df = pd.DataFrame({
                "Aminoácido": aa_labels,
                "Repeticiones": aa_values,
                "Porcentaje": aa_percentages
            })

            fig = px.pie(aa_df, names='Aminoácido', values='Porcentaje', hover_data={'Aminoácido': True, 'Porcentaje': True},
                         title="Distribución de Aminoácidos", 
                         labels={"Porcentaje": "Porcentaje (%)", "Aminoácido": "Aminoácido"})
            fig.update_traces(textinfo='percent+label', hovertemplate='%{label}: %{value}%')
            st.subheader("Distribución de Aminoácidos")
            st.plotly_chart(fig)

        if show_sequence_alignment:
            st.write("Puedes realizar un alineamiento de secuencias usando herramientas como [Clustal Omega](https://www.ebi.ac.uk/Tools/msa/clustalo/) o [MAFFT](https://mafft.cbrc.jp/alignment/software/)")
            alignment_file = st.file_uploader("Sube tu archivo para el alineamiento de secuencias (FASTA)", type=["fasta", "fa"])
            if alignment_file is not None:
                try:
                    with io.StringIO(alignment_file.getvalue().decode("utf-8")) as alignment_stringio:
                        alignment_record = SeqIO.read(alignment_stringio, "fasta")
                    st.write(f"Secuencia cargada para el alineamiento: {alignment_record.id}")
                    st.text_area("Secuencia cargada", str(alignment_record.seq), height=200)
                except Exception as e:
                    st.error(f"Error al leer el archivo de alineamiento: {e}")

        if show_secondary_structure:
            st.write("Puedes predecir la estructura secundaria de la proteína usando herramientas como [PSIPRED](https://www.ebi.ac.uk/Tools/psipred/).")
            secondary_structure_file = st.file_uploader("Sube tu archivo para la predicción de la estructura secundaria (FASTA)", type=["fasta", "fa"])
            if secondary_structure_file is not None:
                try:
                    with io.StringIO(secondary_structure_file.getvalue().decode("utf-8")) as secondary_structure_stringio:
                        secondary_structure_record = SeqIO.read(secondary_structure_stringio, "fasta")
                    st.write(f"Secuencia cargada para predicción de estructura secundaria: {secondary_structure_record.id}")
                    st.text_area("Secuencia cargada", str(secondary_structure_record.seq), height=200)
                except Exception as e:
                    st.error(f"Error al leer el archivo de estructura secundaria: {e}")

        if show_3d_and_factor_b:
            st.write("Para generar un modelo 3D a partir de tu secuencia de aminoácidos, puedes utilizar herramientas como I-TASSER o AlphaFold.")
            st.write("Una vez generado el modelo 3D, puedes cargar el archivo PDB aquí para visualizarlo y analizar el factor B.")
            st.write("Puedes obtener archivos PDB desde sitios como [Protein Data Bank](https://www.rcsb.org/) o [ModBase](https://modbase.compbio.ucsf.edu/).")

            # Subir archivo PDB solo si se selecciona el checkbox "Modelo 3D y Factor B"
            pdb_file = st.file_uploader("Sube tu archivo PDB para ver el modelo 3D y el factor B", type=["pdb"])

            if pdb_file is not None:
                try:
                    # Confirmando detalles del archivo subido
                    st.write(f"Archivo PDB subido: {pdb_file.name}")
                    st.write(f"Tamaño del archivo: {len(pdb_file.getvalue())} bytes")

                    # Crear archivo temporal para almacenar el contenido del PDB
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp_file:
                        tmp_file.write(pdb_file.getvalue())
                        tmp_file_path = tmp_file.name
                        st.write(f"Archivo PDB temporal creado en: {tmp_file_path}")

                    # Verifica que el archivo PDB es válido
                    parser = PDBParser(QUIET=True)
                    structure = parser.get_structure("protein", tmp_file_path)

                    # Extraer las coordenadas de los átomos y calcular el factor B (aquí, se usa un valor aleatorio como ejemplo)
                    atoms = []
                    residues = []
                    factor_b_values = []

                    for model in structure:
                        for chain in model:
                            for residue in chain:
                                factor_b = np.random.random()  # Valor aleatorio como ejemplo del factor B
                                factor_b_values.append(factor_b)
                                for atom in residue:
                                    atoms.append({
                                        "element_symbol": atom.element,
                                        "x_coord": atom.coord[0],
                                        "y_coord": atom.coord[1],
                                        "z_coord": atom.coord[2],
                                        "residue_name": residue.get_resname()
                                    })

                    # Crear un DataFrame de las coordenadas de los átomos
                    df_atom = pd.DataFrame(atoms)
                    df_factor_b = pd.DataFrame({"Residuo": [residue.get_resname() for residue in structure.get_residues()],
                                                "Factor_B": factor_b_values})

                    # Crear gráfico 2D de número de residuo vs. factor B al cuadrado
                    df_factor_b["Residuo"] = np.arange(1, len(df_factor_b) + 1)  # Asignar número de residuo
                    df_factor_b["Factor_B_Squared"] = df_factor_b["Factor_B"] ** 2

                    fig_factor_b = px.line(df_factor_b, x="Residuo", y="Factor_B_Squared", 
                                           title="Número de Residuo vs Factor B²",
                                           labels={"Residuo": "Número de Residuo", "Factor_B_Squared": "Factor B²"})
                    st.subheader("Gráfico de Número de Residuo vs Factor B²")
                    st.plotly_chart(fig_factor_b)

                    # Visualización 3D de los átomos usando Plotly
                    fig_3d = px.scatter_3d(df_atom, x="x_coord", y="y_coord", z="z_coord", color="element_symbol", template="plotly_dark",
                                           color_discrete_sequence=["white", "gray", "red", "yellow"])
                    fig_3d.update_coloraxes(showscale=True)
                    fig_3d.update_traces(marker=dict(size=3))

                    st.subheader("Distribución de Atomos en 3D (Color por Elemento)")
                    st.plotly_chart(fig_3d)

                    # Visualización 3D de los residuos usando Plotly (color por residuo)
                    fig_residue = px.scatter_3d(df_atom, x="x_coord", y="y_coord", z="z_coord", color="residue_name", template="plotly_dark")
                    fig_residue.update_traces(marker=dict(size=3))
                    fig_residue.update_coloraxes(showscale=True)

                    st.subheader("Distribución de Aminoácidos en 3D (Color por Residuo)")
                    st.plotly_chart(fig_residue)

                except Exception as e:
                    st.error(f"Error al procesar el archivo PDB: {e}")
                    st.write(f"Error de detalles: {e}")
                finally:
                    # Elimina el archivo temporal después de su uso
                    if os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)
