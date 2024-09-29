from pite import tim_simulator, hubbard_simulator

if __name__ == "__main__":

    
    for step in [1,10,20,30,40]:
    
        Nimrod = tim_simulator( num_ite = step, # time steps
                                num_site = 4,
                                num_shots = 50000,
                                )
        
        
        Nimrod.run() 
    
    """
    for step in [1,10,20,30,40,50,60]: 
        Wyvern = hubbard_simulator( num_ite=step,
                                    num_site=2,    # aka spatial orb
                                    num_shots=25000,
                                    dTau=0.1,
                                    t_val=0.1,
                                    U_val=0.1,
                                    )
        Wyvern.run() 
    """

