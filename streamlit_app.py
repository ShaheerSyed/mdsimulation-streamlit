# Shaheer Syed
# streamlit_app.py
# Inspired by https://github.com/klyshko/md_python/blob/master/Lecture.ipynb


##----------- Importing Necessary Libraries ------------##

import streamlit as st
from streamlit_molstar import st_molstar, st_molstar_rcsb, st_molstar_remote

import requests

import MDAnalysis as md

from openmm.app import *
from openmm import *
from openmm.unit import *

from MDAnalysis.analysis import dihedrals

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import plotly.express as px
import pandas as pd
from sys import stdout

import os

## ---------------------Front End (Main Page)-----------------------------##

# Size of Image
st.image("logo.png", width=1000)

## Description
st.write(
    
    """

**Understanding Molecular Dynamics:**
Molecular dynamics (MD) is a powerful computer simulation method delving into the dynamic evolution of atoms and molecules. In this virtual realm, the trajectories of particles are determined by solving Newton's equations of motion. The forces and potential energies governing these movements are often calculated using molecular mechanics force fields.

Explore the intricate dance of atoms and molecules as you simulate their dynamic behaviors with this application!

""",

 unsafe_allow_html=True)


## Input Message
st.info('⬅ Please Select the Protein of Your Choice and the Temperature to Run the MD Simulation from the Sidebar!')


## ---------------------- Back End (Functions)-----------------------------##

##---------------------- Function to run the simulation--------------------##

def run_simulation(your_protein, temp):
    
    print(f'the name of the chosen protein is {your_protein}')
    
    pdb_filename = f'{your_protein}.pdb'
    pdb_url = f'https://raw.githubusercontent.com/klyshko/md_python/master/data/{pdb_filename}'

    # Download the PDB file
    response = requests.get(pdb_url)

    # Check if the request was successful (status code 200).
    
    if response.status_code == 200:
        with open(pdb_filename, 'wb') as f:  # Use 'wb' for binary write mode
        
            f.write(response.content)
        
        st.success(f"PDB file saved as '{pdb_filename}'.")
    else:
        st.error(f"Failed to fetch the PDB file from {pdb_url}. Status code: {response.status_code}")

    
# Run the simulation using the downloaded PDB file

    
    pdb = PDBFile(pdb_filename)
    ff = ForceField('amber10.xml')
    system = ff.createSystem(pdb.topology, nonbondedMethod=CutoffNonPeriodic)
    temperature = temp
    frictionCoeff = 1/picosecond
    time_step = 0.002*picoseconds
    total_steps = 400*picoseconds / time_step
    integrator = LangevinIntegrator(temperature, frictionCoeff, time_step)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()
    
    simulation.reporters.append(DCDReporter(f'{your_protein}_traj.dcd', 1000))
    
    simulation.reporters.append(StateDataReporter(stdout, 5000, step=True, potentialEnergy=True,\
                                                  temperature=True, progress=True, totalSteps=total_steps))
    simulation.step(total_steps)
    
    st.success(f"Trajectory file saved as '{your_protein}_traj.dcd'.")
    st.success("The Simulation has completed successfully! Please see the results below.")  


## ----------------- Function to visualize the trajectory using Mol* viewer  ----------------- ##


def visualize_trajectory(your_protein):
    
    # Load the trajectory
    traj_file_path = f'{your_protein}_traj.dcd'
    
    
    # Load the PDB file
    pdb_file_path = f'{your_protein}.pdb'
  
    print(f'{traj_file_path} and {pdb_file_path}')
    
    # Display Visual using Mol* component
        
    st_molstar(pdb_file_path, traj_file_path, key="10", height= 600)
    
    

## ----------------- Function to analyze end-to-end distance ----------------- ##

def analyze_end_to_end_distance(your_protein):
    
    sys = md.Universe(f'{your_protein}.pdb', f'{your_protein}_traj.dcd')
    
    N_terminus = sys.select_atoms('resid 1 and name N')
    C_terminus = sys.select_atoms('resid 25 and name C')
    
    dist = []
    timesteps = []
    
    for frame in sys.trajectory:
        timesteps.append(frame.time)
        dist.append(np.linalg.norm(N_terminus.positions - C_terminus.positions))
        
    data = pd.DataFrame({'Timesteps': timesteps, 'End-to-End Distance': dist})
    
  ## Create an interactive plot with Plotly
    fig = px.line(data, x='Timesteps', y='End-to-End Distance', labels={'End-to-End Distance': 'end-to-end distance, Å'})

    #fig.update_layout(
    #title='End-to-End Distance Analysis',
    #xaxis_title='Timesteps',
    #yaxis_title='end-to-end distance, Å'
    #)

    st.plotly_chart(fig)
                    
    #fig.show()
    
    #fig, ax = plt.subplots(figsize=(8, 6))  # Add this line
    
    #px.line(data, x='Timesteps', y='End-to-End Distance', labels={'End-to-End Distance': 'end-to-end distance, Å'}).update_layout(
    #    title='End-to-End Distance Analysis',
    #    xaxis_title='Timesteps',
    #    yaxis_title='end-to-end distance, Å'
    #).show()  # Replace st.plotly_chart with px.line...show()

## ----------------- Function to analyze Ramachandran plot ----------------- ##


def analyze_ramachandran_plot(your_protein):
    sys = md.Universe(f'{your_protein}.pdb', f'{your_protein}_traj.dcd')
    ram1 = dihedrals.Ramachandran(sys).run(0, 30)
    ram2 = dihedrals.Ramachandran(sys).run(170, 200)
    fig, ax = plt.subplots(figsize=(8, 8))
    ram1.plot(ax=ax, color='k', marker='.')
    ram2.plot(ax=ax, color='b', marker='.')
    ax.arrow(20, 20, -40, -40, width=2, head_width=8, head_length=12, fc='b', ec='b')
    ax.text(30, 20, 'alpha region', color='blue', fontsize=20)
    ax.arrow(20, 150, -40, 0, width=2, head_width=8, head_length=12, fc='k', ec='k')
    ax.text(30, 150, 'beta region', fontsize=20)
    st.pyplot(fig)


## ----------------- Function to perform PCA and plot in 2D ----------------- ##

def perform_pca(your_protein):
    sys = md.Universe(f'{your_protein}.pdb', f'{your_protein}_traj.dcd')
    CA_atoms = sys.select_atoms('name CA')
    N = len(CA_atoms)
    M = len(sys.trajectory)
    X = np.empty((M, int(N * (N - 1

) / 2)))
    k = 0
    for frame in sys.trajectory:
        x = []
        for i in range(len(CA_atoms)):
            for j in range(len(CA_atoms)):
                if j > i:
                    d = np.linalg.norm(CA_atoms[i].position - CA_atoms[j].position)
                    x.append(d)
        x = np.array(x)
        X[k] = x
        k += 1
    total_num_components = int(N * (N - 1) / 2)
    n_components = min(total_num_components, M)
    skl_PCA = PCA(n_components=n_components).fit(X)
    num_components = 0
    for i in range(total_num_components):
        total_variance = np.sum(skl_PCA.explained_variance_ratio_[:i])
        if total_variance > 0.90:
            num_components = i
            break
    skl_X_transformed = skl_PCA.transform(X)
    colors = np.array([np.linspace(0, 1, len(skl_X_transformed)), np.linspace(1, 0, len(skl_X_transformed)), np.zeros(len(skl_X_transformed))])
    plt.scatter(skl_X_transformed[:, 0], skl_X_transformed[:, 1], c=colors.T)
    plt.xlabel("PC #1")
    plt.ylabel("PC #2")
    plt.title("Data in 2D")
    st.pyplot(plt)






## ------------------------ Front End (Sidebar) --------------------------------##

## Main Function Logic 

def main():
    
    ## Title Shown on Sidebar 
    
    st.sidebar.title("Protein Molecular Dynamics Simulation App")
    
    st.sidebar.markdown("&copy; Developed by [Shaheer Syed](https://www.linkedin.com/in/shaheer-syed-b9b8221b1/) for CHEM 303 with Dr. Miles")
    
    ## Message Shown on Sidebar
    
    st.sidebar.info("Choose a protein from the options below and specify the temperature for the simulation you'd like to run!")
  

    ## The dropdown box of protein options. 
    
    your_protein = st.sidebar.selectbox("Select Protein", ["polyALA", "polyGLY", "polyGV"])
    
    ## Temperature Selection Slider 
    
    temp = st.sidebar.slider("Select Temperature (K)", min_value=100, max_value=400, step=100, value=100)


    ## Once Run Simulation Button Is Clicked
    
    if st.sidebar.button("Run Simulation"):
        
        with st.spinner("Running Simulation..."):
            run_simulation(f'{your_protein}', temp)
        
        
    
    st.title("Protein Simulation Results")

    st.subheader("1. Visualize Trajectory")
    
    if st.button("Show Trajectory"):
        
        st.warning("To animate the trajectory, follow these steps: \n 1. Click 'Select Animation'. \n 2. Set 'Mode' to 'Once'. \n 3. Click 'Start'. \n 4. Click 'Select Animation' again to close window.")

        visualize_trajectory(f'{your_protein}')
        

    st.subheader("2. Analyze End-to-End Distance")
    
    if st.button("Plot End-to-End Distance"):
       
        analyze_end_to_end_distance(f'{your_protein}')

    st.subheader("3. Analyze Ramachandran Plot")
    
    if st.button("Plot Ramachandran Plot"):
        analyze_ramachandran_plot(f'{your_protein}')

    st.subheader("4. Perform PCA and Plot in 2D")
    
    if st.button("Perform PCA"):
        perform_pca(f'{your_protein}')
        

# --------------------------------------------------------------------------- #
        
if __name__ == "__main__":
    main()
