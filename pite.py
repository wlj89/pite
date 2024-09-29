"""
    Simulation of Non-unitary PITE 
"""

from qiskit import execute, Aer, ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import AncillaQubit, AncillaRegister
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import RXGate
#from qiskit.circuit.gate import RXGate
from qiskit_aer import AerSimulator
from math import acos,exp,sqrt
import numpy as np

import matplotlib

class pite_simulator():
    
    """
        parent simulator class
        child: TIM & Hubbard
    """
    
    def __init__(self,
                 num_ite,
                 num_site,
                 num_shots,
                 dTau,
                 ) -> None:
        
        self.num_ite = num_ite
        self.num_site = num_site
        self.num_shots = num_shots
        self.dTau = dTau

        self.total_successful_shots = 0.0
        self.energy_sum = 0.0 
        self.stddev = 0.0

    def print_psi_qubit(self,state_vec,num_bits):

        #print ("print state vec as it is")
        #print (state_vec)
        
        get_bin = lambda x, n: format(x, 'b').zfill(n)
        
        for idx,val in enumerate(np.asarray(state_vec)):
            print("|",get_bin(idx,num_bits),"> :",val)
    
class hubbard_simulator(pite_simulator):

    """
        hubbard model simulation
    """

    def __init__(self,
                 t_val = 0.5,
                 U_val = 0.1,
                 num_ite = 10,   # num of trotter steps
                 num_site = 2,   # spatial orbital 
                 num_shots = 100000,
                 dTau = 0.1
                 )->None:
        
        pite_simulator.__init__(self,
                                num_ite = num_ite , 
                                num_site = num_site * 2, # spin orb
                                num_shots = num_shots,
                                dTau = dTau,
                                )

        self.num_spatial_orb = num_site
        self.t = t_val   # t > 0 
        self.U = U_val   # u > 0    
        self.total_successful_shots = 0

    def run(self):
        """
            each Pauli string has its own experiment
        """
        print("******SIMULATION PARAMETERS******")

        sim_para = vars(self)
        
        for var in sim_para:
            print(var, "=", sim_para[var])
        

        print("*********************************")    
        
        """
            wave function based simulation
        """
        E_exact = self.simulate_psi()
        
        """ 
            sampling-based simulation
        """
        str_list = []

        base = [] 
        for i in range(self.num_site):
            base.append("I")

        # hopping terms 
        for orb_i in range(self.num_spatial_orb):
            for orb_j in range(orb_i+1,self.num_spatial_orb):
                for spin in [0,1]:
                    i = 2*orb_i + spin
                    j = 2*orb_j + spin
                    
                    # !!!
                    temp = base.copy()
                    
                    # JW
                    temp[i] = "X"
                    for q in range(i+1,j):
                        temp[q] = "Z"
                    temp[j] = "X"

                    str_list.append([temp.copy(),-0.5*self.t])

                    temp = base.copy() 
                    temp[i] = "Y"
                    for q in range(i+1,j):
                        temp[q] = "Z"
                    temp[j] = "Y"

                    str_list.append([temp,-0.5*self.t])

        # repulsion terms 
        for orb in range(self.num_spatial_orb):
            i = 2*orb
            j = 2*orb+1

            temp = base.copy()
            
            temp[i] = "Z"
            temp[j] = "Z"
            str_list.append([temp.copy(), 0.25 * self.U])

            temp = base.copy()
            temp[i] = "Z"
            str_list.append([temp.copy(), -0.25 * self.U])

            temp = base.copy()
            temp[j] = "Z"
            str_list.append([temp.copy(), -0.25 * self.U])

        # Identity string 
        temp = base.copy()
        str_list.append([temp.copy(), 0.5 * self.U])
        
        num_str  = len(str_list)

        for string_n_coeff in str_list:
            print(string_n_coeff)
            self.simulate(string_n_coeff)

        print("SUMMARY")
        print("# Trotter step:",self.num_ite)
        print("<E>_exact =",E_exact)
        print("<E>=",self.energy_sum)
        print("stddev(E)=",sqrt(self.stddev))
        print("rate of success=",self.total_successful_shots/(self.num_shots*num_str)) 
        print("\n")

    def calc_single_config(self, real_bitstr, string)->float:
        """
            all non-I are effectively Z now 
        """
        val = 1 
        
        for idx, op in enumerate(string):
            if not op == "I":
                # watch out the order
                if real_bitstr[self.num_site-1-idx]=='0':
                    val *= 1
                else:
                    val *= -1
        return val                
        
    def calc_expectation(self,result_dict,string_n_coeff)->float:
        
        string = string_n_coeff[0]
        coeff = string_n_coeff[1]

        #print("computing Pauli string ",string)
        #print("coeff: ",coeff)

        var = 0.0
        energy = 0.0; 
        num_successful_shots = 0 
        
        for bitstr in result_dict:
            # singling out successful shots
            # Most significant bit (MSB) as the ancilla
            # print ("actual:",bitstr)
            if bitstr[0] == '0':
                
                real_bitstr = bitstr[1:]
                
                print(real_bitstr, result_dict[bitstr])    
                
                num_successful_shots += result_dict[bitstr]
                # not U yet 
                energy +=  self.calc_single_config(real_bitstr,string) * result_dict[bitstr]
                
        # <h_i>
        energy/=num_successful_shots

        self.total_successful_shots += num_successful_shots

        #print ("energy of string ",string, coeff * energy)
        
        # calc variance 
        for bitstr in result_dict:
            if bitstr[0] == '0':
                
                real_bitstr = bitstr[1:]

                var += result_dict[bitstr] * (self.calc_single_config(real_bitstr,string) - energy)**2

        var/=num_successful_shots
        
        #print ("variance of string ",string, var)

        # confusing divider. Why do we need it
        self.energy_sum += coeff * energy
        
        self.stddev += (coeff)**2 * var / num_successful_shots

    def compute_E(self,psi_0):
        
        H = np.array([[0,0,-self.t,-self.t],
                     [0,0, self.t, self.t],
                     [-self.t, self.t, self.U,0],
                     [-self.t, self.t, 0,self.U]])

        phi = np.matmul(H,psi_0)

        return np.inner(phi.conjugate(),psi_0)        

    def simulate_psi(self):
        """
            state vector simulation
            More code resue could be better...
        """
        from math import pi 
        # final qubit/bit as ancilla 
        
        Qubits= QuantumRegister(1+self.num_site)
        Clbits = ClassicalRegister(1+self.num_site)    

        qc = QuantumCircuit(Qubits,Clbits)
        # also take into account the sign from -Z
        
        get_phi = lambda gamma : 2*acos(exp(-2*abs(gamma)*self.dTau))  

        """
            prepare trial wave function 
            the singlet state 
        """
        qc.h(qc.qubits[0])
        qc.ry(pi,qc.qubits[0])

        qc.x(qc.qubits[1])

        qc.x(qc.qubits[3])

        qc.cx(0,1)
        qc.cx(1,2)
        qc.cx(2,3)
        
        """
            ITE starts
        """
        for ite in range(self.num_ite):
            # hopping terms 
            # orb_idx ~ [0,num_spatial_orb-1]
            for orb_i in range(self.num_spatial_orb):
                for orb_j in range(orb_i+1,self.num_spatial_orb):
                    for spin in [0,1]:
                        
                        # site number in the spin chain 
                        
                        i = 2*orb_i+spin
                        j = 2*orb_j+spin
                        
                        # coefficient of the string 
                        gamma = -self.t * 0.5  # < 0 , no X needed 
                        
                        """
                            ...XZ...ZX...
                        """
                        qc.h(j).c_if(Clbits,0)
                        qc.h(i).c_if(Clbits,0)
                        
                        # CNOT ladder
                        for idx in range(i,j):
                            qc.cx(idx,idx+1).c_if(Clbits,0)

                        # alpha = -t/2, no X needed
                        qc.crx(get_phi(gamma), j, self.num_site).c_if(Clbits,0)

                        # enforce non-unitary evolution
                        qc.measure(qc.qubits[self.num_site],qc.clbits[self.num_site]).c_if(Clbits,0)

                        # CNOT ladder: back 
                        for idx in range(j-1, i-1, -1):
                            qc.cx(idx,idx+1).c_if(Clbits,0)

                        qc.h(j).c_if(Clbits,0)
                        qc.h(i).c_if(Clbits,0)

                        """
                            ...YZ...ZY...
                        """
                        
                        qc.p(-pi/2,i).c_if(Clbits,0)
                        qc.h(i).c_if(Clbits,0)

                        qc.p(-pi/2,j).c_if(Clbits,0)
                        qc.h(j).c_if(Clbits,0)

                        # CNOT ladder
                        for idx in range(i,j):
                            qc.cx(idx,idx+1).c_if(Clbits,0)

                        # no X needed 
                        qc.crx(get_phi(gamma), j, self.num_site).c_if(Clbits,0)

                        # non-unitry
                        qc.measure(qc.qubits[self.num_site],qc.clbits[self.num_site]).c_if(Clbits,0)

                        # CNOT ladder: back 
                        for idx in range(j-1,i-1,-1):
                            qc.cx(idx,idx+1).c_if(Clbits,0)

                        # put things back
                        qc.h(i).c_if(Clbits,0)
                        qc.p(pi/2,i).c_if(Clbits,0)

                        qc.h(j).c_if(Clbits,0)
                        qc.p(pi/2,j).c_if(Clbits,0)
                                    
            # repulsion term
            for site in range(self.num_spatial_orb):
                
                gamma = self.U * 0.25 # > 0, need X 

                #  0.25 *[...I * (I-Z)_{i,up} * I ... I * (I-Z)_{i,down} * I ...]
                #  0.25 I I 
                # -0.25 I Z
                # -0.25 Z I 
                #  0.25 Z Z
                i = 2*site       # up 
                j = 2*site+1     # down
                
                # single Z up 
                # extra minus sign, no x needed
                #qc.x(i).c_if(Clbits,0)
                qc.crx(get_phi(gamma), i, self.num_site).c_if(Clbits,0)
                qc.measure(qc.qubits[self.num_site],qc.clbits[self.num_site]).c_if(Clbits,0)
                #qc.x(i).c_if(Clbits,0)
                
                # single Z down
                # extra minus sign, no X needed
                #qc.x(j).c_if(Clbits,0)
                qc.crx(get_phi(gamma), j, self.num_site).c_if(Clbits,0)    
                qc.measure(qc.qubits[self.num_site],qc.clbits[self.num_site]).c_if(Clbits,0)
                #qc.x(j).c_if(Clbits,0)
                
                # double Z
                # need X 
                qc.cx(i,j).c_if(Clbits,0)

                qc.x(j).c_if(Clbits,0)

                qc.crx(get_phi(gamma), j, self.num_site).c_if(Clbits,0)
                qc.measure(qc.qubits[self.num_site],qc.clbits[self.num_site]).c_if(Clbits,0)
                
                qc.x(j).c_if(Clbits,0)

                qc.cx(i,j).c_if(Clbits,0)
                
                # identity string plays no role: just an global phase
        #end circuit 

        while(1):
            backend = Aer.get_backend("statevector_simulator")
            
            result = execute(experiments=qc, backend=backend, shots=1).result()
            
            psi = result.get_statevector(experiment = qc) 
            
            mk = True
            # check if the shot is successful
            for idx, val in enumerate(psi):
                if idx >= (1<<self.num_site) and abs(val) > 1E-15:
                    mk = False
                    break 
            
            print("shot succesful?",mk)
            
            self.print_psi_qubit(result.get_statevector(experiment = qc),self.num_site+1)   

            if mk is True:
                """
                    compute exact <psi|H|psi>
                """
                print("computing exact expecation value")
                
                psi_0 = np.array([psi[9],psi[6],psi[12],psi[3]])

                E = self.compute_E(psi_0)

                print("<E> from exact Trotterized ITE",E)
                break      
        return E
            
    def simulate(self,string_n_coeff)->None:
        
        from math import pi 
        
        string = string_n_coeff[0]

        # final qubit/bit as ancilla 
        
        Qubits= QuantumRegister(1+self.num_site)
        Clbits = ClassicalRegister(1+self.num_site)    

        qc = QuantumCircuit(Qubits,Clbits)
        # also take into account the sign from -Z
        
        get_phi = lambda gamma : 2*acos(exp(-2*abs(gamma)*self.dTau))  

        """
            prepare trial wave function 
            the singlet state 
        """
        qc.h(qc.qubits[0])
        qc.ry(pi,qc.qubits[0])

        qc.x(qc.qubits[1])

        qc.x(qc.qubits[3])

        qc.cx(0,1)
        qc.cx(1,2)
        qc.cx(2,3)
        
        """
            ITE starts
        """

        for ite in range(self.num_ite):
            # hopping terms 
            # orb_idx ~ [0,num_spatial_orb-1]
            for orb_i in range(self.num_spatial_orb):
                for orb_j in range(orb_i+1,self.num_spatial_orb):
                    for spin in [0,1]:
                        
                        # site number in the spin chain 
                        
                        i = 2*orb_i+spin
                        j = 2*orb_j+spin
                        
                        # coefficient of the string 
                        gamma = -self.t * 0.5  # < 0 , no X needed 
                        
                        """
                            ...XZ...ZX...
                        """
                        qc.h(j).c_if(Clbits,0)
                        qc.h(i).c_if(Clbits,0)
                        
                        # CNOT ladder
                        for idx in range(i,j):
                            qc.cx(idx,idx+1).c_if(Clbits,0)

                        # alpha = -t/2, no X needed
                        qc.crx(get_phi(gamma), j, self.num_site).c_if(Clbits,0)

                        # enforce non-unitary evolution
                        qc.measure(qc.qubits[self.num_site],qc.clbits[self.num_site]).c_if(Clbits,0)

                        # CNOT ladder: back 
                        for idx in range(j-1, i-1, -1):
                            qc.cx(idx,idx+1).c_if(Clbits,0)

                        qc.h(j).c_if(Clbits,0)
                        qc.h(i).c_if(Clbits,0)

                        """
                            ...YZ...ZY...
                        """
                        
                        qc.p(-pi/2,i).c_if(Clbits,0)
                        qc.h(i).c_if(Clbits,0)

                        qc.p(-pi/2,j).c_if(Clbits,0)
                        qc.h(j).c_if(Clbits,0)

                        # CNOT ladder
                        for idx in range(i,j):
                            qc.cx(idx,idx+1).c_if(Clbits,0)

                        # no X needed 
                        qc.crx(get_phi(gamma), j, self.num_site).c_if(Clbits,0)

                        # non-unitry
                        qc.measure(qc.qubits[self.num_site],qc.clbits[self.num_site]).c_if(Clbits,0)

                        # CNOT ladder: back 
                        for idx in range(j-1,i-1,-1):
                            qc.cx(idx,idx+1).c_if(Clbits,0)

                        # put things back
                        qc.h(i).c_if(Clbits,0)
                        qc.p(pi/2,i).c_if(Clbits,0)

                        qc.h(j).c_if(Clbits,0)
                        qc.p(pi/2,j).c_if(Clbits,0)
                                    
            # repulsion term
            for site in range(self.num_spatial_orb):
                
                gamma = self.U * 0.25 # > 0, need X 

                #  0.25 *[...I * (I-Z)_{i,up} * I ... I * (I-Z)_{i,down} * I ...]
                #  0.25 I I 
                # -0.25 I Z
                # -0.25 Z I 
                #  0.25 Z Z
                i = 2*site       # up 
                j = 2*site+1     # down
                
                # single Z up 
                # extra minus sign, no x needed
                #qc.x(i).c_if(Clbits,0)
                qc.crx(get_phi(gamma), i, self.num_site).c_if(Clbits,0)
                qc.measure(qc.qubits[self.num_site],qc.clbits[self.num_site]).c_if(Clbits,0)
                #qc.x(i).c_if(Clbits,0)
                
                # single Z down
                # extra minus sign, no X needed
                #qc.x(j).c_if(Clbits,0)
                qc.crx(get_phi(gamma), j, self.num_site).c_if(Clbits,0)    
                qc.measure(qc.qubits[self.num_site],qc.clbits[self.num_site]).c_if(Clbits,0)
                #qc.x(j).c_if(Clbits,0)
                
                # double Z
                # need X 
                qc.cx(i,j).c_if(Clbits,0)

                qc.x(j).c_if(Clbits,0)

                qc.crx(get_phi(gamma), j, self.num_site).c_if(Clbits,0)
                qc.measure(qc.qubits[self.num_site],qc.clbits[self.num_site]).c_if(Clbits,0)
                
                qc.x(j).c_if(Clbits,0)

                qc.cx(i,j).c_if(Clbits,0)
                
                # identity string plays no role: just an global phase

        #end circuit 

        # apply unitary transformation if needed 
        for i in range(self.num_site):
            if string[i] == "X":
                # H^\dagger 
                qc.h(i).c_if(Clbits,0)
            if string[i] == "Y":
                # S^\dagger H 
                qc.p(-pi/2,i).c_if(Clbits,0)
                qc.h(i).c_if(Clbits,0)            
        
        for i in range(self.num_site):
            qc.measure(qc.qubits[i],qc.clbits[i]) 
        
        #qc.draw(output="mpl", filename="ite.png")
        #return 
    
        result = execute(experiments=qc, backend=AerSimulator(),shots=self.num_shots).result()
        
        # measurement record. 
        #bit_str = result.get_memory(experiment=qc)
        self.calc_expectation(result.get_counts(),string_n_coeff)


class tim_simulator(pite_simulator):
    """
        transverse Ising model simulation
    """
    def __init__(self,
                 J_val = 0.5,
                 h_val = 0.1,
                 num_ite = 10,   # num of trotter steps
                 num_site = 4,
                 num_shots = 100000,
                 dTau = 0.1
                 ):
        
        pite_simulator.__init__(self,
                                num_ite = num_ite,
                                num_site = num_site,
                                num_shots = num_shots,
                                dTau = dTau,
                                )
        
        self.J = J_val 
        self.h = h_val

        if self.J < 0:
            raise Exception("J value smaller than 0 not accepted")
        if self.h < 0:
            raise Exception("h value smaller than 0 not accepted")


    def run(self):
        """
            each Pauli string has its own experiment
            
        """
        print("******SIMULATION PARAMETERS******")

        sim_para = vars(self)
        
        for var in sim_para:
            print(var, "=", sim_para[var])
        
        print("*********************************")    
        
        """
            wave function based caculation 
        """
        
        E_exact = self.simulate_psi()

        """
            sampling based calculation
        """
        # string plus coefficient
        str_list = [] 
        
        base = [] 
        for i in range(self.num_site):
            base.append("I")
        
        for i in range(self.num_site):
            next_neigh_idx = (i+1)%self.num_site
            temp = base[:]
            temp[i]="Z"
            temp[next_neigh_idx]="Z"
            str_list.append([temp,-self.J])
        
        for i in range(self.num_site):
            temp=base[:]
            temp[i]="X"
            str_list.append([temp,-self.h])

        for string_n_coeff in str_list:
            print(string_n_coeff)
            self.simulate(string_n_coeff)

        num_str = len(str_list)
        
        print("SUMMARY")
        print("num of step:",self.num_ite)
        print("<E>=",self.energy_sum)
        print("<E>_exact=",E_exact)
        print("stddev(E)=",sqrt(self.stddev))
        print("rate of success=", self.total_successful_shots / (num_str * self.num_shots))
        #self.calc_expectation()

    def compute_E(self,psi):
        
        get_bin = lambda x, n: format(x, 'b').zfill(n)

        get_pos = lambda x: self.num_site - 1 - x

        phi = np.array([0+0j for i in range(len(psi))])

        for idx_bin, val in enumerate(psi):
             
            idx_str = get_bin(idx_bin,self.num_site)

            # ZZ, diagonal
            for i in range(0,self.num_site):

                i_next = (i+1)%self.num_site 
                
                if idx_str[get_pos(i)] == idx_str[get_pos(i_next)]:
                    phi[idx_bin] += (-self.J)*val
                else:
                    phi[idx_bin] += self.J * val
                
            # X, off diag 
            for i in range(0,self.num_site): 
                
                i_real = get_pos(i)
                if idx_bin & (1 << i_real) == (1 << i_real):
                    # 1 bit, becomes 0 after X 
                    phi[idx_bin - (1 << i_real)] += -self.h * val
                else: 
                    # 0 -> 1 
                    phi[idx_bin + (1 << i_real)] += -self.h * val
        
        
        return np.inner(psi.conjugate(), phi)
                        
        
    def simulate_psi(self):
        from math import pi

        # quantum registor. final bit as ancilla
        Qubits= QuantumRegister(1+self.num_site)
        
        # classical register. final bit as ancilla -> the MSB of bit-string representation
        Clbits = ClassicalRegister(1+self.num_site)    

        qc = QuantumCircuit(Qubits,Clbits)

        # initialize the trial wf
        qc.h([i for i in range(self.num_site)])

        phi_J = 2 * acos(exp(-2*abs(self.J)*self.dTau)) 
        phi_h = 2 * acos(exp(-2*abs(self.h)*self.dTau)) 

        # ITE with num_ite steps
        for ite in range(self.num_ite):    
            # applying -J*ZZ
            for i in range(0,self.num_site):

                next_neigh_idx = (i+1)%self.num_site
                
                # parity accumulation through CX ladder
                # c_if branch: only precede if classical register is 0 (might save time?)
                # that is, unitary op is successful with ancilla measure to be 0
                # clreg is 0 when spawned, so doesn't affect the first iteration
                qc.cx(i,next_neigh_idx).c_if(Clbits,0)

                # flipping when gamma > 0 
                # alpha  = -J, no X needed 
                #qc.x(next_neigh_idx).c_if(Clbits,0)

                # controlled RX on the ancilla qubit 
                qc.crx(phi_J,next_neigh_idx,self.num_site).c_if(Clbits,0)    

                #if J > 0: 
                #qc.x(next_neigh_idx).c_if(Clbits,0)

                # measure the ancilla
                qc.measure(qc.qubits[self.num_site],qc.clbits[self.num_site]).c_if(Clbits,0)

                # put things back
                qc.cx(i,next_neigh_idx).c_if(Clbits,0)

            # applying -h*X
            
            for i in range(0,self.num_site):
                # H-gate unitarily transforms X to Z 
                qc.h(i).c_if(Clbits,0)

                #qc.x(i).c_if(Clbits,0)
                qc.crx(phi_h, i, self.num_site).c_if(Clbits,0)
                   
                #print(num_site)
                qc.measure(qc.qubits[self.num_site],qc.clbits[self.num_site]).c_if(Clbits,0)

                # put things back
                #qc.x(i).c_if(Clbits,0)
                qc.h(i).c_if(Clbits,0)      

        while(1):
            backend = Aer.get_backend("statevector_simulator")
            
            result = execute(experiments=qc, backend=backend, shots=1).result()
            
            psi = result.get_statevector(experiment = qc) 
            
            mk = True
            # check if the shot is successful
            for idx, val in enumerate(psi):
                if idx >= (1<<self.num_site) and abs(val) > 1E-15:
                    mk = False
                    break 
            
            print("shot succesful?",mk)
            
            self.print_psi_qubit(result.get_statevector(experiment = qc),self.num_site+1)   

            if mk is True:
                #break
                """
                    compute exact <psi|H|psi>
                """
                print("computing exact expecation value")
                
                #psi_0 = np.array([psi[9],psi[6],psi[12],psi[3]])

                E = self.compute_E(psi)

                print("<E> from exact Trotterized ITE",E)

                return E    
        
        

    def simulate(self,string_n_coeff):
        
        from math import pi
        
        string = string_n_coeff[0]

        # quantum registor. final bit as ancilla
        Qubits= QuantumRegister(1+self.num_site)
        
        # classical register. final bit as ancilla -> the MSB of bit-string representation
        Clbits = ClassicalRegister(1+self.num_site)    

        qc = QuantumCircuit(Qubits,Clbits)

        # initialize the trial wf
        qc.h([i for i in range(self.num_site)])

        phi_J = 2 * acos(exp(-2*abs(self.J)*self.dTau)) 
        phi_h = 2 * acos(exp(-2*abs(self.h)*self.dTau)) 

        # ITE with num_ite steps
        for ite in range(self.num_ite):    
            # applying -J*ZZ
            for i in range(0,self.num_site):

                next_neigh_idx = (i+1)%self.num_site
                
                # parity accumulation through CX ladder
                # c_if branch: only precede if classical register is 0 (might save time?)
                # that is, unitary op is successful with ancilla measure to be 0
                # clreg is 0 when spawned, so doesn't affect the first iteration
                qc.cx(i,next_neigh_idx).c_if(Clbits,0)

                # flipping when gamma > 0 
                # alpha  = -J, no X needed 
                #qc.x(next_neigh_idx).c_if(Clbits,0)

                # controlled RX on the ancilla qubit 
                qc.crx(phi_J,next_neigh_idx,self.num_site).c_if(Clbits,0)    

                #if J > 0: 
                #qc.x(next_neigh_idx).c_if(Clbits,0)

                # measure the ancilla
                qc.measure(qc.qubits[self.num_site],qc.clbits[self.num_site]).c_if(Clbits,0)

                # put things back
                qc.cx(i,next_neigh_idx).c_if(Clbits,0)

            # applying -h*X
            
            for i in range(0,self.num_site):
                # H-gate unitarily transforms X to Z 
                qc.h(i).c_if(Clbits,0)

                #qc.x(i).c_if(Clbits,0)
                qc.crx(phi_h, i, self.num_site).c_if(Clbits,0)
                   
                #print(num_site)
                qc.measure(qc.qubits[self.num_site],qc.clbits[self.num_site]).c_if(Clbits,0)

                # put things back
                #qc.x(i).c_if(Clbits,0)
                qc.h(i).c_if(Clbits,0)      
            
        # apply unitary transformation if needed
        for i in range(self.num_site):
            if string[i] == "X":
                # H^\dagger 
                qc.h(i).c_if(Clbits,0)
            if string[i] == "Y":
                qc.p(-pi/2,i).c_if(Clbits,0)
                qc.h(i).c_if(Clbits,0)       
        
        for i in range(self.num_site):
            qc.measure(qc.qubits[i],qc.clbits[i]) 
        #qc.draw(output="mpl", filename="ite.png")

        # have to record measurements at each shot 
        result = execute(experiments=qc, backend=AerSimulator(),shots=self.num_shots).result()
        
        # measurement record. 
        #bit_str = result.get_memory(experiment=qc)
        self.calc_expectation(result.get_counts(),string_n_coeff)

    # end simulate() 

    def calc_one_config(self,real_bitstr,string)->float:
        # all non-I are effectively Z now
        val = 1 
        
        for idx, op in enumerate(string):
            if not op == "I":
                # watch out the order
                if real_bitstr[self.num_site-1-idx]=='0':
                    val *= 1
                else:
                    val *= -1
        return val
    
            
    def calc_expectation(self,result_dict,string_n_coeff):
        
        string = string_n_coeff[0]
        coeff = string_n_coeff[1]

        print("computing Pauli string ",string)
        print("coeff:",coeff)
        
        var = 0.0
        energy = 0.0; 
        num_successful_shots = 0 
        
        for bitstr in result_dict:
            # singling out successful shots
            # Most significant bit (MSB) as the ancilla
            if bitstr[0] == '0':
                
                real_bitstr = bitstr[1:]
                
                print(real_bitstr, result_dict[bitstr])    
                
                num_successful_shots += result_dict[bitstr]
                # not -J or -h yet 
                energy +=  self.calc_one_config(real_bitstr,string) * result_dict[bitstr]

        # <h_i>
        energy/=num_successful_shots
        
        self.total_successful_shots += num_successful_shots
        
        print ("energy of string ",string, coeff * energy)
                
        for bitstr in result_dict:
            if bitstr[0] == '0':
                
                real_bitstr = bitstr[1:]

                var += result_dict[bitstr] * (self.calc_one_config(real_bitstr,string) - energy)**2

        var/=num_successful_shots
        
        print ("variance of string ",string, var)
        
        # confusing divider. Why do we need it
        self.energy_sum += coeff * energy
        self.stddev += (coeff**2) * var / num_successful_shots


        
if __name__ == "__main__":
    #state_vec_test()
    #Z_test()
    #X_test() 
    #ITE_TIM(num_site=4,num_ite=20)
    
    Nimrod = tim_simulator(num_ite = 30, 
                           num_site = 8,
                           num_shots = 100000,
                           )
    
    Nimrod.run() 