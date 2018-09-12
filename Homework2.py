
# coding: utf-8

import sys
try:
    import docplex.mp
except: 
    if hasattr(sys, 'real_prefix'):
        get_ipython().system('pip install docplex')
    else:
        get_ipython().system('pip install --user docplex')
from docplex.mp.model import Model
import numpy as np
import timeit

#Reads the file and saves all the parameters is a list named parameters
def read_file(file_name):
    parameters=[]
    with open(file_name) as f:
        lines=f.read().splitlines()
        for line in lines:
            parameters.append([float(value) for value in line.split()])
    return parameters

#Constructs dictionaries to save all the parameters of the model
def set_parameters(parameters): 
    
    num_variables=int(parameters[0][0])
    num_constraints=int(parameters[0][1])
    num_cons_ineq=int(parameters[0][2])
    
    #Saves the cost coefficients in a dictionary
    costs_Objfunc1={}
    costs_Objfunc2={}
    for i in range(num_variables):
        costs_Objfunc1.update({i:parameters[1][i]})
        costs_Objfunc2.update({i:parameters[2][i]})
    
    #Saves the right hand side of the constraints in a dictionary
    right_hand={}
    for i in range(num_constraints):
        right_hand.update({i:parameters[3+num_constraints][i]})
    
   #Saves the constraint coefficients in a dictionary
    coef_const={}
    for i in range(3,num_constraints+3):
        for j in range(num_variables):
            coef_const.update({(i-3,j):parameters[i][j]})

    return num_variables, num_constraints, num_cons_ineq, costs_Objfunc1, costs_Objfunc2, right_hand, coef_const

#Constructs the model based on the parameters 
def construct_model(num_variables, num_constraints, num_cons_ineq, right_hand, coef_const): 
    
    #Create one model instance
    m = Model()

    #Define the decision variables
    x={i:m.integer_var(name='x_{0}'.format(i)) for i in range(num_variables)} 

    #Define the constraints
    #Constrainst with inequality 
    for i in range(num_cons_ineq):
        m.add_constraint( m.sum(x[j]*coef_const.get((i,j)) for j in range(num_variables))<= right_hand[i])

    #Constrainst with equality 
    for i in range(num_cons_ineq,num_constraints):
        m.add_constraint( m.sum(x[j]*coef_const.get((i,j)) for j in range(num_variables))== right_hand[i])
    
    return m, x

#Solves the model and returns the optimal solution and their image in the criterion space 
def solve_model(m, x, num_variables, costs_Objfunc): 
    
    #Solves the instance created under the name of m
    s = m.solve()
    
    #If the problem is infeasible
    if s is None:
        opt_solution=None
        Objfunc_value=None
    
    #If the problem is feasible
    else: 
        #Dictionary with the value of the optimal solution 
        opt_solution={}
        opt_solution.update({i:s[x[i]] for i in range(num_variables)})
    
        #Value of the objective function in the optimal solution
        Objfunc_value=np.sum(costs_Objfunc[i]*opt_solution[i] for i in range(num_variables))
    
    return opt_solution, Objfunc_value 

#Initializes the model using the lexicographic operation
def initialize_model(num_variables, num_constraints, num_cons_ineq, right_hand, coef_const, costs_Objfunc1, costs_Objfunc2): 
    epsilon=0.0001

    #Solve for Objective function 1 first
    
    #Constructs an object for the model
    m, x=construct_model(num_variables, num_constraints, num_cons_ineq, right_hand, coef_const)
    #Adds the objective function 1 to the object m
    m.minimize(m.sum(x[i]*costs_Objfunc1.get((i)) for i in range(num_variables))) 
    #Solves the object m
    opt_solution1, Objfunc_value1= solve_model(m, x, num_variables, costs_Objfunc1)

    #Solve for Objective function 2 adding the constraint of the value of the Objective function 1
    #Removes the Objective function 1 
    m.remove_objective
    #Adds the constraint that makes the Objective Function 1 equal to the value obtained previously
    m.add_constraint(m.sum(x[i]*costs_Objfunc1.get((i)) for i in range(num_variables))<= Objfunc_value1+epsilon, ctname="cons_OF")
    #Adds the objective function 2 to the object m
    m.minimize(m.sum(x[i]*costs_Objfunc2.get((i)) for i in range(num_variables))) 
    #Solves the object m
    opt_solution2, Objfunc_value2= solve_model(m, x, num_variables, costs_Objfunc2)
    #Saves the top extreme non supported point
    z_top={0:Objfunc_value1,1: Objfunc_value2}

    
    #Solve for Objective function 2 first
    
    #Removes the constraint of the value of the Objective function 1
    m.remove_constraint("cons_OF")
    #Solves the object m
    opt_solution21, Objfunc_value21= solve_model(m, x, num_variables, costs_Objfunc2)

    #Solve for Objective function 1 adding the constraint of the value of the Objective function 2
    #Removes the Objective function 2 
    m.remove_objective
    #Adds the constraint that makes the Objective Function 2 equal to the value obtained in the previous model
    m.add_constraint(m.sum(x[i]*costs_Objfunc2.get((i)) for i in range(num_variables))<= Objfunc_value21+epsilon)
    #Adds the objective function 1 to the object m
    m.minimize(m.sum(x[i]*costs_Objfunc1.get((i)) for i in range(num_variables))) 
    #Solves the object m
    opt_solution22, Objfunc_value22= solve_model(m, x, num_variables, costs_Objfunc1)
    #Saves the bottom extreme non supported point
    z_bottom={0:Objfunc_value22,1: Objfunc_value21}  

    return z_top, opt_solution2, z_bottom, opt_solution22 

#Applies the perpendicular search method and returns a list with the non-dominated points and non-dominated solutions
def perpendicular_search(z_top, z_bottom, opt_solution1, opt_solution2, num_variables, num_constraints, num_cons_ineq, right_hand, coef_const, costs_Objfunc1, costs_Objfunc2): 
    epsilon=0.99
    
    # Creates a list for the non dominated points
    nondominated_points=[]
    nondominated_points.extend([z_top, z_bottom])
    
    # Creates a list for the non dominated solutions corresponding to the non dominated points obtained
    nondominated_solutions=[]
    nondominated_solutions.extend([opt_solution1, opt_solution2])
    
    point_queue=[]
    point_queue.append((z_top,z_bottom))

    # Constructs an object from the model
    m, x=construct_model(num_variables, num_constraints, num_cons_ineq, right_hand, coef_const)
    # Adds the objective function with lambda1=lambda2=1
    m.minimize(m.sum(x[i]*costs_Objfunc1.get((i))for i in range(num_variables))+m.sum(x[i]*costs_Objfunc2.get((i))for i in range(num_variables))) 
    #Counts iterations
    iteration=1
    
    
    while len(point_queue)!=0:
    
        #Assigns the values of the non dominated points in the point_queue list
        z_old=point_queue.pop(0)
        z11_old=z_old[:1][0][0]
        z12_old=z_old[:1][0][1]
        z21_old=z_old[:2][1][0]
        z22_old=z_old[:2][1][1]
        z1_old=z_old[:1][0]
        z2_old=z_old[:2][1]
        
        #If it is not the first iteration, remove the old contrainst with the values of the objective functions
        if iteration !=1: 
            m.remove_constraint("cons_OF1")
            m.remove_constraint("cons_OF2")
    
        #Add contrainsts with the maximum value of the objective functions
        m.add_constraint(m.sum(x[i]*costs_Objfunc1.get((i)) for i in range(num_variables))<= z21_old-epsilon, ctname="cons_OF1")
        m.add_constraint(m.sum(x[i]*costs_Objfunc2.get((i)) for i in range(num_variables))<= z12_old-epsilon, ctname="cons_OF2")
        
        #Solves the problem
        optimal_solution, z1_new= solve_model(m, x, num_variables, costs_Objfunc1)
        
        #If the problem is feasible
        if optimal_solution != None: 
        
            #Calculates the value for the Objective Function 2
            z2_new=np.sum(costs_Objfunc2[i]*optimal_solution[i] for i in range(num_variables))
            #Assigns the value of the new non dominated point
            z_new={0:z1_new, 1:z2_new}
            #Adds the new "squares" in the list
            point_queue.extend([(z1_old,z_new),(z_new,z2_old)])
            #Adds the extreme non dominated points found
            nondominated_points.append(z_new)
            #Adds the non dominated solutions
            nondominated_solutions.append(optimal_solution)
    
    return nondominated_points, nondominated_solutions 

# Writes the output file
def write_output(nondominated_solutions, nondominated_points, elapsed, num_variables):
    
    #Creates the output file
    file = open("problem_solutions.txt", "w")
    #Writes the non dominated points, non dominated solutions and run time in the text file
    file.write("Nondominated points")
    for i in range(1,len(nondominated_points)+1): 
        file.write("\n"+str(i)+" Objective Function 1= "+str(nondominated_points[i-1][0])+" Objective Function 2= "+str(nondominated_points[i-1][1]))
    
    file.write("\n"+ "Nondominated solutions")
    for i in range(1,len(nondominated_solutions)+1):
        file.write("\n"+str(i))
        for j in range(1,num_variables+1):
                   file.write(" X"+str(j)+"= "+str(nondominated_solutions[i-1][j-1]))
    
    file.write("\n"+"Run Time= "+str(elapsed)+" s")
                    
    file.close()

if __name__ == "__main__":
    
    start_time = timeit.default_timer()
    file_name='parameters.txt'
    #Reads the input file
    parameters= read_file(file_name)

    #Sets the values of the parameters of the model
    num_variables, num_constraints, num_cons_ineq, costs_Objfunc1, costs_Objfunc2, right_hand, coef_const=set_parameters(parameters)

    #Initialize the model by computing the end points
    z_top, opt_solution1, z_bottom, opt_solution2=initialize_model(num_variables, num_constraints, num_cons_ineq, right_hand, coef_const, costs_Objfunc1, costs_Objfunc2)

    #Applies the Perpendicular Search Method and returns the results
    nondominated_points, nondominated_solutions=perpendicular_search(z_top, z_bottom, opt_solution1, opt_solution2, num_variables, num_constraints, num_cons_ineq, right_hand, coef_const, costs_Objfunc1, costs_Objfunc2)

    elapsed = timeit.default_timer() - start_time

    #Writes de Output file
    write_output(nondominated_solutions, nondominated_points, elapsed, num_variables)

