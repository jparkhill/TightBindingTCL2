#include "TCLprop.h"
using namespace std;
using namespace arma;

<<<<<<< HEAD
int main()
{
    int norb = 10;
    
    cout << "norb: " << norb << endl;
    cout << "Beta: " << Beta << endl;
    
    cx_mat H = BuildLinearPolyene(norb);
    arma::cx_mat Rho(norb,norb),Rho_(norb,norb);
    arma::cx_mat NewRho(norb,norb);
    Rho.zeros(); NewRho.zeros();
    cx_vec eigs_unsrt;
    arma::cx_mat Cunsrt;
	arma::eig_gen(eigs_unsrt,Cunsrt,H);
    uvec ord = sort_index(real(eigs_unsrt));
    cx_vec eigs(eigs_unsrt(ord));
    cx_mat C(Cunsrt.cols(ord));
//    C.print("C: ");
    
    real(eigs.t()).print("Energies: ");
=======
using namespace std; 
using namespace arma; 


//Dynamics parameters:
int nstates = 25;
int ntimes = 5;//5000
int norb = 25;
int noc = 84;// ?
int ini_ex = 8;
double dt_aimd = 1.0;// fs
double dt_elec = 0.5; // au?
>>>>>>> 68f92cfe0e866195d932a6912b35595060406857

    // Verify that C transforms H into a diagonal form.
    //cx_mat tmp(C.t()*H*(C));
    //tmp.print("Diagonalized?");// yes.
    
    TCLMatrices MyTCL(norb, C, eigs);

<<<<<<< HEAD
    int ntimesAu = int(ntimes/(FsPerAu*10)+1);
    int loop = 0; int loop2 =0;
  
    cx_mat Rho0 = diagmat(FermiDirac(norb/2,eigs,Beta));
    Rho = Rho0;
    
    cx_vec Boltz = Boltzmann(norb/2,eigs,Beta); //Get a normalized boltzmann distribution
 //    Rho = diagmat(Boltzmann(norb/2,eigs,Beta)); Without blocking, canonical detailed balance.
    real(Boltz.t()).print("BoltzPops: ");
    real(Rho.diag().t()).print("FdPops: ");

// Equilibrate from the Ground State
//    Rho = Gs(norb/2,eigs);
    
// Equilibrate from An Excited State
    Rho = Es(norb/2,eigs,2);
    
// Equilibrate from left filling.
    Rho = Left(norb/2,eigs,Beta,C);
	
//	Rho = CoherentEven(norb/2,eigs,Beta,C);
//	Rho = CoherentLeft(norb/2,eigs,Beta,C);
	
// Equilibrate from infinite temperature.
//     Rho = diagmat(Infinite(norb/2,eigs,Beta));// Note this makes the real space density antidiagonal zero.
=======
void LRhoMinusRRho(double tnow, double dt_aimd, cx_mat& OldRho, cx_mat& NewRho){
		std::complex<double> ii(0.0,1.0);
//		MyTCL.ContractGammaRho(NewRho,OldRho);
//		NewRho.print("ContractGammaRho returns ");
//		arma::cx_mat HatT = Hts.HatT(tnow,dt_aimd);
//		HatT.print("HatT returns");
//		NewRho += -ii*(HatT*OldRho - OldRho*HatT);
		//ContractGammaRho Returns the action of -K*rho NOTE THE MINUS SIGN IS ALREADY included.
}
  
  // Takes OldRho = Density Matrix now.
  // Gives DM at t+dt puts it in NewRho

void RK4Step_bothterms(arma::cx_mat& OldRho,arma::cx_mat& NewRho, double tnow=0.0, double dt=dt_elec, double dt_aimd=dt_aimd){
	arma::cx_mat k1(nstates,nstates),k2(nstates,nstates),k3(nstates,nstates),k4(nstates,nstates),v2(nstates,nstates),v3(nstates,nstates),v4(nstates,nstates);
	LRhoMinusRRho(tnow, dt_aimd,OldRho,k1); // dRho/dt = -i/hbar*(H*Rho - Rho*H) - R*Rho
		//cout << "PRINTING k1" << endl;
		//k1.print();
//		v2 = (dt/2) * k1;
//        v2 += OldRho;
//        
//        LRhoMinusRRho(tnow+(dt/2),dt_aimd,v2,k2);
//	    v3 = (dt/2) * k2;
//        v3 += OldRho;
//        
//        LRhoMinusRRho(tnow+(dt/2),dt_aimd,v3,k3);
//        v4 = (dt) * k3;
//        v4 += OldRho;
//        
//        LRhoMinusRRho(tnow+dt,dt_aimd,v4,k4);
//		
//        NewRho = OldRho;
//        NewRho += (1.0/6.0)*k1;
//        NewRho += (2.0/6.0)*k2;
//        NewRho += (2.0/6.0)*k3;
//        NewRho += (1.0/6.0)*k4;
}

int main()
{
//	cout << "Hello World" << endl;
//	int No = 4;
//	int Nocc = 2;  
//	arma::mat C(4,4),S(4,4),F(4,4);
//	double eigs[4]; 
//	TCLMatrices Test(No,C,S,F,eigs,Nocc); 
//	cout << "Constructed the test" << endl;
>>>>>>> 68f92cfe0e866195d932a6912b35595060406857

	cout <<"Running dynamics using RK-45..." <<  endl;
	double dt = dt_elec; // This is just the initial timestep
	double stepmax = dt_elec*10.0;
	double tolmax = pow(10.0,-5.0);
	double t=0;
	const long double wallt0 = time(0); // This is in wall-seconds.
	double WallHrPerPs=0;

	while(true)
	{
		// Because of the variable timestep this won't always work. That's okay it works well enough. ;)
		if (fmod(t*FsPerAu,100.0)<(0.55*dt*FsPerAu) or loop%50==0)
		{
			Rho_ = rTransform(Rho,C);
            cout << " loop: " << loop << endl;
            cout << " Trace of This Rho: " << trace(Rho) << endl;
            real(Rho.diag().t()).print("Eigenspace populations: ");
            real(Rho_.diag().t()).print("Real Space populations: ");
			abs(Rho_).print("Rho-Rs");
			cout << "P_gs: " << P_gs(Rho) << endl;
            // the antidiagonal of the density at the center.
            real(Rho0.diag().t()).print("FdPops: ");
			loop2 +=1;
		}
		//RK4(MyTCL, diagmat(eigs), Rho, NewRho, t, dt);
		RK45(MyTCL, diagmat(eigs), Rho, NewRho, stepmax, t, dt, tolmax);
		Rho=NewRho;
        Rho.print("Rho");
		WallHrPerPs = ((time(0)-wallt0)/(60.0*60.0))/(t*FsPerAu/1000.0);
		cout << loop << " t: "<< t*FsPerAu << "(fs) - dt " << dt*FsPerAu <<"(fs) - WallHrPerPs: " << WallHrPerPs << " - Elapsed: " << (int)(time(0)-wallt0) << " (s) - Trace(Rho): " << trace(Rho) << endl;
		// If you're within dt of 100fs save the populations to a file on disk.
		loop +=1;
	}

<<<<<<< HEAD
=======
cout <<"Running dynamics..." <<  endl;
//for (double t=0; t<(ntimes-1)/FsPerAu; t+=dt_elec){
for (double t=0; t<0.5; t+=dt_elec){
	//Interpolate H:
	if (t==0){Rho = Rho0;/*Rho0.print(); This looks right*/}
	RK4Step_bothterms(Rho,NewRho,t,dt_elec, dt_aimd);
	Rho= NewRho;
//	Rho.print();
}

	return 0;
>>>>>>> 68f92cfe0e866195d932a6912b35595060406857
}
