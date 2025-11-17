# Imports
from pgmpy.models import BayesianNetwork, DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import State
from pgmpy.sampling import BayesianModelSampling
from sympy.strategies.core import switch

#from bayesnet import evidence


# ---------Create nodes and conditional probabilityh distribution ------

def get_probs_gene_ancestor(varName):
    """
    Builds a conditional probability distribution (TabularCPD)
    using a priori probabilities, for a variable gene with no
    parents

    Parameter
    ----------
    varName : name of the variable (String)
    """
    if not isinstance(varName, str):
        raise TypeError("The name of the variable must be a string")

    probs_gene = TabularCPD (
        variable=varName,
        variable_card=3,
        values=[ [0.01], # 2
                 [0.03], # 1
                 [0.96] # 0
        ],
        state_names={varName: ["2","1","0"]}
    )
    return probs_gene



def get_probs_trait(varName,evidenceName):
    """
    Builds a conditional probability distribution (TabularCPD)
    for traits given the number of genes

    Parameters
    ----------
    varName : name of the traits variable (String)
    evidenceName: name of the evidence gene variable (String)
    """
    if not (isinstance(varName, str) and isinstance(evidenceName, str)):
        raise TypeError("The name of the variables must be a string")

    probs_trait =TabularCPD (
        variable= varName,
        variable_card=2,
        values=[[0.65, 0.56, 0.01], # oui
                [0.35, 0.44, 0.99] # non
        ],
        evidence=[evidenceName],
        evidence_card=[3],
        state_names={varName: ["oui","non"],
                     evidenceName: ["2","1","0"]}
    )
    return probs_trait



# constant defining mutation probability of a gene
prob_mutation = 0.01

def get_probs_heredity1(geneParent):
    """
    Computes probability of inheriting 1 gene from
    a given parent:
    P(Gene_{inherited chromosome}|Gene_{parent})

    Parameter
    ----------
    geneParent: number of genes (0, 1 or 2) of the
    parent (father or mother)
    """
    if geneParent == 0:
        return prob_mutation
    elif geneParent == 1:
        return 0.5
    elif geneParent == 2:
        return 1 - prob_mutation

    else:
        raise ValueError("Invalid value for geneParent")



def get_probs_gene(varNameChild,evidenceNameFather,evidenceNameMother):
    """
    Builds a conditional probability distribution (TabularCPD)
    for the number of genes of a child given the number of genes
    of each of the parents

    Parameters
    ----------
    varNameChild : name of the traits variable (String)
    evidenceName: name of the evidence gene variable (String)
    """
    if not (isinstance(varNameChild, str) and isinstance(evidenceNameFather, str) and isinstance(evidenceNameMother, str)):
        raise TypeError("The name of the variables must be a string")

    values_child2 = []
    values_child1 = []
    values_child0 = []

    for geneFather in [2,1,0]:
        for geneMother in [2,1,0]:
            p = get_probs_heredity1(geneFather)
            m = get_probs_heredity1(geneMother)

            prob2 = p * m
            prob1 = p * (1-m) + (1-p) * m
            prob0 = (1-p)*(1-m)

            values_child2.append(prob2)
            values_child1.append(prob1)
            values_child0.append(prob0)



    probs_gene = TabularCPD (
        variable=varNameChild,
        variable_card= 3,
        values=[
            values_child2, # Gene_child = 2 => proba = p * m
            values_child1, # Gene_child = 1 => proba = gets_probs_heridity1(p) * (1-gets_probs_heridity1(m)) + (1-gets_probs_heridity1(p)) * gets_probs_heridity1(m)
            values_child0  # Gene_child = 0 => proba = (1 - p)*(1 - m)
        ],
        evidence=[evidenceNameFather, evidenceNameMother],
        evidence_card=[3,3],
        state_names={varNameChild: ["2","1","0"],
                     evidenceNameFather: ["2","1","0"],
                     evidenceNameMother: ["2","1","0"]}
    )

    return probs_gene

#  +----------------+-------+-----+-----+-----+-----+-----+-----+-----+-----+
#  | Gene_Father    |   2   |  2  |  2  |  1  |  1  |  0  |  0  |  0  |  0  |
#  +----------------+-------+-----+-----+-----+-----+-----+-----+-----+-----+
#  | Gene_Mother    |   2   |  1  |  0  |  2  |  1  |  0  |  2  |  1  |  0  |
#  +----------------+-------+-----+-----+-----+-----+-----+-----+-----+-----+
#  | Gene_Child=2   |0.9801 |0.495|             ???                           |
#  +----------------+-------+-----+-----+-----+-----+-----+-----+-----+-----+
#  | Gene_Child=1   |                       ???                           |
#  +----------------+-------+-----+-----+-----+-----+-----+-----+-----+-----+
#  | Gene_Child=0   |                       ???                           |
#  +----------------+-------+-----+-----+-----+-----+-----+-----+-----+-----+


#--------------------Create a Bayesian Network for family n°1 --------------------------------

model1 = DiscreteBayesianNetwork(
    [('Gene_Leto', 'Gene_Paul'),
    ('Gene_Jessica', 'Gene_Paul'),
     ('Gene_Leto', 'Gene_Alia'),
     ('Gene_Jessica', 'Gene_Alia'),
     ('Gene_Leto', 'Trait_Leto'),
     ('Gene_Jessica', 'Trait_Jessica'),
     ('Gene_Alia', 'Trait_Alia'),
     ('Gene_Paul', 'Trait_Paul')]
)

model1.add_cpds(get_probs_gene_ancestor('Gene_Leto'),
                get_probs_gene_ancestor('Gene_Jessica'),
                get_probs_gene('Gene_Paul','Gene_Leto','Gene_Jessica'),
                get_probs_gene('Gene_Alia','Gene_Leto','Gene_Jessica'),
                get_probs_trait('Trait_Leto','Gene_Leto'),
                get_probs_trait('Trait_Jessica','Gene_Jessica'),
                get_probs_trait('Trait_Alia','Gene_Alia'),
                get_probs_trait('Trait_Paul','Gene_Paul'))
model1.check_model()

viz = model1.to_graphviz()
viz.draw('family1.png', prog='dot')


#--------------------Inference for family n°1 --------------------------------

# Exact inference 
infer = VariableElimination(model1)



# Calculate predictions based on the evidence provided by the Trait variables
result1 = infer.query(variables=["Gene_Paul"],
                     evidence= {"Trait_Leto" : 'non', "Trait_Jessica" : 'oui', "Trait_Alia" : 'oui', "Trait_Paul" : 'non'})
print(result1)


# Calculate predictions based on the evidence provided by the Trait variables and knowing that Jessica and Alia have resp. 1 and 2 genes

result2 = infer.query(variables=["Gene_Paul"],
                     evidence= {"Trait_Leto" : 'non', "Trait_Jessica" : 'oui', "Trait_Alia" : 'oui', "Trait_Paul" : 'non', "Gene_Jessica" : '1', "Gene_Alia" : '2'})
print(result2)


# Approximate inference
sampler = BayesianModelSampling(model1)
evidence = [
    State(var='Trait_Leto', state='non'),
    State(var='Trait_Jessica', state='oui'),
    State(var='Trait_Alia', state='oui'),
    State(var='Trait_Paul', state='non')
]
samples = sampler.rejection_sample(evidence=evidence, size=5000)
counts = samples['Gene_Paul'].value_counts(normalize=True)

print("Distribution approchée de Gene_Paul :")
print(counts)

#--------------------Create a Bayesian Network for family n°2 --------------------------------

lstP = ['Charles', 'Diana', 'Michael', 'Carole', 'Harry', 'Meghan', 'William', 'Katherine', 'Philippa',
        'Archie', 'Liliet', 'George', 'Charlotte', 'Louis']


model2 = DiscreteBayesianNetwork(
    [
        ('Charles', 'Harry'),
        ('Charles', 'William'),
        ('Diana', 'Harry'),
        ('Diana', 'William'),
        ('Harry', 'Archie'),
        ('Harry', 'Liliet'),
        ('Meghan', 'Archie'),
        ('Meghan', 'Liliet'),
        ('Michael', 'Katherine'),
        ('Michael', 'Philippa'),
        ('Carole', 'Philippa'),
        ('Carole', 'Katherine'),
        ('William', 'George'),
        ('William', 'Charlotte'),
        ('William', 'Louis'),
        ('Katherine', 'George'),
        ('Katherine', 'Charlotte'),
        ('Katherine', 'Louis'),
        ('Charles', 'T_Charles'),
        ('Diana', 'T_Diana'),
        ('Michael', 'T_Michael'),
        ('Carole', 'T_Carole'),
        ('Harry', 'T_Harry'),
        ('Meghan', 'T_Meghan'),
        ('William', 'T_William'),
        ('Katherine', 'T_Katherine'),
        ('Philippa', 'T_Philippa'),
        ('Archie', 'T_Archie'),
        ('Liliet', 'T_Liliet'),
        ('George', 'T_George'),
        ('Charlotte', 'T_Charlotte'),
        ('Louis', 'T_Louis')
    ]
)

model2.add_cpds(
    get_probs_gene_ancestor('Charles'),
    get_probs_gene_ancestor('Diana'),
    get_probs_gene_ancestor('Michael'),
    get_probs_gene_ancestor('Carole'),
    get_probs_gene_ancestor('Meghan'),
    get_probs_gene('Harry','Charles','Diana'),
    get_probs_gene('William','Charles','Diana'),
    get_probs_gene('Archie', 'Harry', 'Meghan'),
    get_probs_gene('Liliet', 'Harry', 'Meghan'),
    get_probs_gene('Philippa', 'Michael', 'Carole'),
    get_probs_gene('Katherine', 'Michael', 'Carole'),
    get_probs_gene('George', 'William', 'Katherine'),
    get_probs_gene('Charlotte', 'William', 'Katherine'),
    get_probs_gene('Louis', 'William', 'Katherine'),
    get_probs_trait('T_Charles','Charles'),
    get_probs_trait('T_Diana','Diana'),
    get_probs_trait('T_Michael','Michael'),
    get_probs_trait('T_Carole','Carole'),
    get_probs_trait('T_Harry','Harry'),
    get_probs_trait('T_Meghan','Meghan'),
    get_probs_trait('T_William','William'),
    get_probs_trait('T_Katherine','Katherine'),
    get_probs_trait('T_Philippa','Philippa'),
    get_probs_trait('T_Archie','Archie'),
    get_probs_trait('T_Liliet','Liliet'),
    get_probs_trait('T_George','George'),
    get_probs_trait('T_Charlotte','Charlotte'),
    get_probs_trait('T_Louis','Louis')
)

model2.check_model()

viz = model2.to_graphviz()
viz.draw('family2.png', prog='dot')

#--------------------Inference for family n°2 --------------------------------

# Exact inference
infer = VariableElimination(model2)
dicT = {'T_Charles': 'oui', 'T_Diana': 'non', 'T_Michael': 'non', 'T_Carole': 'non',
        'T_Harry': 'non', 'T_Meghan': 'non', 'T_William': 'non', 'T_Katherine': 'non', 'T_Philippa': 'oui',
        'T_Archie': 'oui', 'T_Liliet': 'non', 'T_George': 'non', 'T_Charlotte': 'non', 'T_Louis': 'oui'}

# Calculate predictions based on the evidence provided by the Trait variables
result_george = infer.query(variables=['George'], evidence=dicT)
result_liliet = infer.query(variables=['Liliet'], evidence=dicT)

print("Probabilité du gène de George :")
print(result_george)

print("\nProbabilité du gène de Liliet :")
print(result_liliet)


# Calculate predictions based on the evidence provided by the Trait variables and knowing that Meghan has no gene

dicT2 = dicT.copy()
dicT2['Katherine'] = "0"
result_george2 = infer.query(variables=['George'], evidence=dicT2)
result_liliet2 = infer.query(variables=['Liliet'], evidence=dicT2)

print("Probabilité du gène de George en sachant en plus que Katherine n’a aucune version défectueuse du gène :")
print(result_george2)

print("\nProbabilité du gène de Liliet en sachant en plus que Katherine n’a aucune version défectueuse du gène :")
print(result_liliet2)
