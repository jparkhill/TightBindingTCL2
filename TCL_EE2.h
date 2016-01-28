#ifndef TCLEE2h
#define TCLEE2h
// <Loggins> DANGERZONEEE...<\Loggins>
#ifndef ARMA_NO_DEBUG
#define ARMA_NO_DEBUG 1
#endif
#ifndef ARMA_CHECK
#define ARMA_CHECK 0
#endif

// Must have user defined reductions which were a feature of OMP 4.0 in 2013.
#ifdef _OPENMP
#if _OPENMP > 201300
#define HASUDR 1
#else
#define HASUDR 0
#endif
#else
#define HASUDR 0
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <armadillo>
#include "math.h"
#include "rks.h"
#include "TCL.h"
#include "TCL_polematrices.h"
// To grab our 2e integrals.
#include "../cdman/rimp2grad.h"
#include "../cdman/ricispd.h"
//#include "noci_mp2.h"
//#include "KeyVtoM.hh"
#include <liblas/liblas.h>
#include "fileman_util.h"

using namespace arma;

class EE2{
public:
	int n_ao, n_mo, n_occ, n_aux;
    std::map<std::string,double> params;
	int ActiveFirst, ActiveLast; // Mostly to avoid rapidly oscillating terms. NYI.
    bool B_Built;
    double nuclear_energy,Ehf,Epseudo,lambda,Emp2; // Lambda is an adiabatic parameter to warm the system.
    cube BpqR; // Lowdin basis, calculated once on construction and fixed throughout.

    mat RSinv;
    mat H,X;
    cx_mat V,C;
    vec eigs;
    
    // This is handy to reduce B rebuilds, if V=Curr then CurrInt is not rebuilt
    cx_mat IntCur, VCur;
    
    typedef complex<double> cx;
    cx j; // The imaginary unit.
	
    EE2(int n_mo_, int n_occ_, rks* the_scf, cx_mat& Rho_, cx_mat& V_,const std::map<std::string,double>& params_ = std::map<std::string,double>(), int ActiveFirst_=0, int ActiveLast_=0): n_mo(n_mo_), j(0.0,1.0), n_occ(n_occ_), H(the_scf->H), X(the_scf->X), V(V_), B_Built(false), params(params_), lambda(1.0)
	{
		int LenV = megtot();
		int IBASIS = rem_read(REM_IBASIS);
		int stat = STAT_NBASIS;
		int Job;
		int IBasisAux = rem_read(REM_AUX_BASIS);
        nuclear_energy = the_scf->nuclear_energy();
        
		ftnshlsstats(&n_aux,&stat, &IBasisAux);
		cout << "------------------------------" << endl;
		cout << "-------- EE2 JAP 2015 --------" << endl;
		cout << "--------   Corr On    --------" << endl;
		cout << "------------------------------" << endl;
        cout << "RIFock:" << params["RIFock"] << endl;
#ifdef _OPENMP
        cout << "OMP spec" << _OPENMP << " n_threads " << omp_get_num_threads() << endl;
#if HASUDR
        cout << "Your compiler supports parallism in EE2. You'll be toasty warm soon." << endl;
#else
        cout << "Your compiler does not support OMP parallism in EE2, please recompile with GCC 5.0 or Intel 16.0 or later" << endl;
#endif
#endif
        
		cout << "NMO: " << n_mo << " NAux: " << n_aux << " Size BpqR " << n_mo*n_mo*n_aux*8/(1024*1024) << "MB" << endl;
		BpqR.resize(n_mo,n_mo,n_aux);
		RSinv.resize(n_aux,n_aux);
		{
			// get (P|Q)**(-1/2) and save it to disk
			//
			//  First, fool trnjob into thinking we want RIMP2
			int tmp1 = rem_read(REM_LEVCOR);
			rem_write(103, REM_LEVCOR);
			trnjob(&Job);
			//  Then, write it back
			rem_write(tmp1, REM_LEVCOR);
			// form (P|Q)^-1/2 matrix and saves it to FILE_2C2E_INV_INTS
			//
			threading_policy::enable_blas_only();
			invsqrtpq(&n_aux, &Job);
			threading_policy::pop();
			// read (P|Q)^-1/2
			FileMan(FM_OPEN_RW, FILE_2C2E_INVSQ_INTS, 0, 0, 0, 0, 0);
			LongFileMan(FM_READ, FILE_2C2E_INVSQ_INTS, FM_DP, n_aux*n_aux, 0, FM_BEG, RSinv.memptr(), 0);
			FileMan(FM_CLOSE, FILE_2C2E_INVSQ_INTS, 0, 0, 0, 0, 0);
		}
		//RSinv.print("RSinv");
        BuildB();
        UpdateB(V_);
        Rho_.eye(); Rho_.submat(n_occ,n_occ,n_mo-1,n_mo-1) *= 0.0;
        
        // To run external to Q-Chem and avoid the two minute link.
        if (params["DumpEE2"])
        {
            V_.save("V",arma_binary);
            X.save("X",arma_binary);
            BpqR.save("BpqR",arma_binary);
        }
        
        // Fock build, integral build and initialize the density.
        CorrelateInitial(Rho_,V_);
	}
    
    // Try to adiabatically turn on the correlation.
    void GML(arma::cx_mat& Rho_, arma::cx_mat& V_)
    {
        cout << "Adiabatically Correlating the system." << endl;
        double dt=0.02; int NGML=300;
        cx_mat RhoM12_(Rho_);
        for(int i=0; i<NGML; ++i)
        {
            lambda=(1.0+i)/((double)NGML);
            cx_mat newrho(Rho_);
            cx_mat Rot = FockBuild(Rho_, V_); // updates eigs, C, V, F, etc. etc. etc.
            //        Rot.print("Rot");
            
            Rho_ = Rot.t()*Rho_*Rot;
            RhoM12_ = Rot.t()*RhoM12_*Rot;
            
            // Make the exponential propagator.
            arma::cx_mat F(Rho_); F.zeros(); F.set_real(diagmat(eigs));
            vec Gd; cx_mat Cu;
            eig_sym(Gd,Cu,F);
            // Full step RhoM12 to make new RhoM12.
            cx_mat NewRhoM12(Rho_);
            cx_mat NewRho(Rho_);
            Split_RK4_Step_MMUT(Gd, Cu, RhoM12_, NewRhoM12, 0.0, dt);
            //        NewRhoM12.print("NewRhoM12");
            // Half step that to make the new Rho.
            Split_RK4_Step_MMUT(Gd, Cu, NewRhoM12, NewRho, 0.0, dt/2.0);
                
            Rho_ = 0.5*(NewRho+NewRho.t());//0.5*(NewRho+NewRho.t());
            RhoM12_ = 0.5*(NewRhoM12+NewRhoM12.t());//0.5*(NewRhoM12+NewRhoM12.t());
            
            if (i%10==0)
                cout << "Gell-Man Low Iteration:" << i << " Lambda " << lambda << " ehf:" << Ehf << endl;
        }
    }
    

    // Generate rate matrix for populations
    void Stability(cx_mat& Rho)
    {
        cout << "calculating stability matrix... " << endl;
        cx_mat tore(Rho); tore.eye();
        cx_mat ref(Rho); ref.zeros();
        double dn=0.001;
        RhoDot(Rho,ref,0.0);
        int n=n_mo;
        for (int j=0; j<n; ++j)
        {
            cx_mat tmp(Rho);
            tmp(j,j)+=dn;
            cx_mat tmp2(Rho); tmp2.zeros();
            RhoDot(tmp,tmp2,0.0);
            for (int i=0; i<n; ++i)
            {
                tore(i,j) += (tmp2(i,i)-ref(i,i))/dn;
            }
        }
        tore.print("Rate Matrix:");
        cout << "Rate Trace? " << trace(tore) << endl;
        for (int j=0; j<n; ++j)
            cout << "Normed Probability:" << j << " " << accu(tore.col(j)) << endl;
        cx_mat sstates; cx_vec rates;
        eig_gen(rates, sstates, tore);
        rates.print("Rates:");
        cout << "State sum: "<< accu(sstates.row(n-1)) << endl;
        sstates.row(n-1).print("steady state");
    }
    
    // Obtain the initial state for the dynamics.
    void CorrelateInitial(cx_mat& Rho,cx_mat& V_)
    {
        cout << "Solving the RI-HF problem to ensure self-consistency." << endl;
        double err=100.0;
        int maxit = 400; int it=1;
        while (abs(err) > pow(10.0,-10) && it<maxit)
        {
            cx_mat Rot = FockBuild(Rho,V_);
            Rho = Rot.t()*Rho*Rot;
            Rho.diag() *= 0.0;
            err=accu(abs(Rho));
            Rho.eye(); Rho.submat(n_occ,n_occ,n_mo-1,n_mo-1)*=0.0;
            if (it%10==0)
                cout << " Fock-Roothan: " << it << std::setprecision(9) << " Ehf" << Ehf << endl;
            it++;
        }
        cx wks0 = eigs(n_occ)-eigs(n_occ-1);
        // Various options for non-idempotent initial state:
        // gell-man-low, fermi-dirac, and mp2NO.
        cx_mat Pmp2=MP2(Rho); // Won't actually solve for Pmp2 unless we want it. Otherwise just gets the CE.
        if ((bool)params["GML"] && !params["Mp2NOGuess"])
            GML(Rho,V_);
        lambda = 1.0;
        if (params["Mp2NOGuess"])
        {
            cout << "Generating MP2 1 particle density matrix..." << endl;
            if (params["Mp2NOGuess"]==2.0)
                Rho = diagmat(Pmp2);
            else
                Rho = Pmp2;
        }
        if (params["FD"])
        {
            vec pops(eigs.n_elem);
            double EvPerAu=27.2113;
            double Kb = 8.61734315e-5/EvPerAu;
            double Beta = 1.0/(params["Temp"]*Kb);
            cout << "Finding " << n_occ*2 << " Fermions at Beta " << Beta << endl;
            double mu = eigs(n_occ);
            
            mu -= 0.010;
            pops = 1.0/(1.0+exp(Beta*(eigs-mu)));
            double sm = sum(pops);
            mu += 0.020;
            pops = 1.0/(1.0+exp(Beta*(eigs-mu)));
            double sp = sum(pops);
            // Linear fit for mu
            mu = eigs(n_occ)-0.01 + ((0.02)/(sp-sm))*(n_occ - sm);
            pops = 1.0/(1.0+exp(Beta*(eigs-mu)));
            cout << "Populated with " << sum(pops) << " Fermions at mu = " << mu <<  endl;
            
            //Repeat
            double dmu=0.001;
            
            while(abs(accu(pops)-n_occ)>pow(10.0,-10.0))
            {
                mu -= dmu;
                pops = 1.0/(1.0+exp(Beta*(eigs-mu)));
                sm = sum(pops);
                mu += 2*dmu;
                pops = 1.0/(1.0+exp(Beta*(eigs-mu)));
                sp = sum(pops);
                // Linear fit for mu
                mu = mu-2.0*dmu + ((2.0*dmu)/(sp-sm))*(n_occ - sm);
                pops = 1.0/(1.0+exp(Beta*(eigs-mu)));
                cout << "Populated with " << accu(pops) << " Fermions at mu = " << mu <<  endl;
                if ( abs(sum(pops)-n_occ) < pow(10.0,-10.0))
                    break;
                dmu /= max(min(10.0,abs(((2.0*dmu)/(sp-sm))*(n_occ - sm))),1.1);
            }
            
            Rho.diag().zeros();
            for (int i=0; i<n_mo; ++i)
                if( abs(pops(i)) > pow(10.0,-16.0))
                    Rho(i,i) = pops(i);
            cout << "Populated with " << sum(pops) << " Fermions at mu = " << mu <<  endl;
            Rho.diag().st().print("Rho(fd)");
        }
        
        if (params["InitialCondition"]==1)
        {
            cx tmp = Rho(n_occ-1,n_occ-1);
            Rho(n_occ,n_occ)+=tmp;
            Rho(n_occ-1,n_occ-1)*=0.0;
        }
        cout << "Initial Trace: " << trace(Rho) << endl;
        
        eigs.st().print("Eigs Before Rebuild");
        FockBuild(Rho,V_);
        UpdateB(V_);
        eigs.st().print("Eigs After Rebuild");
        cx wks1 = eigs(n_occ)-eigs(n_occ-1);
        
        // Print the initial Step, Density, Noccs, and DNoccs.
        {
            Rho.diag().st().print("Inital Rho: ");
            vec noccs; cx_mat u;
            if(!eig_sym(noccs, u, Rho, "std"))
            {
                cout << "Warning... initial density did not diagonalize..." << endl;
                return;
            }
            noccs.st().print("initial natural occupations");
            
            // Check to see what effect the nonlinearity has...
            cx_mat tmp(Rho),tmp2(Rho),tmp3(Rho); tmp.zeros(); tmp3.zeros();
            tmp2.diag(1)+=0.0001;
            tmp2.diag(-1)+=0.0001;
            RhoDot(tmp2,tmp,1.0);
            tmp2.diag(1)+=0.002;
            tmp2.diag(-1)+=0.002;
            RhoDot(tmp2,tmp3,3.0);
            
            // Obtain the correction to a particle hole excitation energy.
            cout << "-------------------------" << endl;
            cout << "Omega_hf 0: " << wks0 << " " << 27.2113*wks0 << "eV"<< endl;
            cout << "Omega_c 0: " << wks1 << " " << 27.2113*wks1 << "eV"<< endl;
            cx ehbe(2.0*IntCur((n_occ)*n_mo+(n_occ),(n_occ-1)*n_mo+n_occ-1)-IntCur((n_occ)*n_mo+(n_occ-1),(n_occ)*n_mo+n_occ-1));
            double cr=(imag(tmp(n_occ,n_occ-1)/0.0001));
            cout << "E-H binding energy. " << ehbe << " " << 27.2113*ehbe << "eV"<< endl;
            cout << "GS MP2 Correlation energy. " << Emp2 << " " << 27.2113*Emp2 << "eV"<< endl;
            cout << "CIS Transition energy. " << ehbe << " " << 27.2113*(wks0+ehbe) << "eV"<< endl;
            cout << "EE2 Correction: " << cr << " " << 27.2113*cr<< "eV"<< endl;
            cout << "EE2 Correction (2): " << " " << 27.2113*(imag(tmp3(n_occ,n_occ-1)/0.002))<< "eV"<< endl;
            cout << "-------------------------" << endl;
            cout << " Total Estimated Homo Lumo xsition Energy: " << 27.2113*(cr+wks1+ehbe) << "eV"<< endl;
            cout << " Total Estimated Homo Lumo xsition Energy+MP2: " << 27.2113*(cr+wks1+ehbe+Emp2) << "eV"<< endl;
            cout << "-------------------------" << endl;
            
            vec dnoccs;
            if(!eig_sym(dnoccs, u, Rho+0.02*tmp, "std"))
            {
                cout << "Warning... change in density did not diagonalize..." << endl;
                return;
            }
            dnoccs -= noccs;
            dnoccs.st().print("initial NO change");
            tmp.print("Initial RhoDot");
        }
        lambda = 1.0;
        UpdateB(V_);
        return;
    }
    
    // Finds the Chemical Potential, and then the best effective temperature to obtain pops.
    // Based on the current eigs to obtain as close to pops as possible.
    double EffectiveTemperature(const vec& pops)
    {
        if (n_occ+2 >= n_mo or n_occ-2<=0)
            return 0.0;
        cout << "Finding effective Temperature..."  << endl;
        double EvPerAu=27.2113;
        double Kb = 8.61734315e-5/EvPerAu;
        vec reigs = real(eigs);
        vec spops = sort(pops, "descend");
        double dmu = pow(10.0,-3.0); double mui = reigs(n_occ-2); double muf = reigs(n_occ+2); // Assume the chemical potential is bw h-2,l+2
        double hldiff =reigs(n_occ) - reigs(n_occ-1);
        double dB = 10.0*hldiff;
        double Bi=dB; double Bf=100.0/hldiff;
        
        double BestMu = 0.0;
        double BestTemp = 0.0;
        double BestError = n_occ;
        
        for(double mu=mui; mu<muf; mu+=dmu)
        {
            for(double Beta=Bi; Beta<Bf; Beta+=dB)
            {
                vec p = pow(exp(Beta*(reigs-mu))+1.0,-1.0);
                double Error = accu((spops-p)%(spops-p));
                if (Error != Error)
                    Error = 8.0;
                else if (Error < BestError)
                {
                    BestTemp=1.0/(Beta*Kb);
                    BestMu=mu;
                    BestError = Error;
                }
            }
        }
        double Beta = 1.0/(BestTemp*Kb);
        cout << "Effective Temperature: " << BestTemp << " at mu= " << BestMu << " beta= " << Beta << " Fermi-Deviance " << sqrt(BestError) << endl;
        vec p = pow(exp(Beta*(reigs-BestMu))+1.0,-1.0);
        (p-spops).st().print("Deviation From Thermality.");
        return BestTemp;
    }
    
    // Calculate the MP2 energy and opdm blockwise in the fock eigenbasis to avoid singles.
    // Use this as a density guess for the dynamics.
    // Since HF is probably very far from the desired initial state
    cx_mat MP2(cx_mat& RhoHF)
    {
        int n = n_mo;
        int no=n_occ;
        int nv=n_mo-n_occ;
        typedef std::complex<double> cx;
        
        int n1=nv;
        int n2=nv*n1;
        int n3=n2*no;
        int n4=no*n3;
        std::vector<cx> Vs(n4),T(n4);
        std::fill(Vs.begin(), Vs.end(), std::complex<double>(0.0,0.0));
        std::fill(T.begin(), T.end(), std::complex<double>(0.0,0.0));

        cx_mat d2 = Delta2();
        for(int R=0; R<n_aux; ++R)
        {
            cx_mat B = V.t()*BpqR.slice(R)*V;
            for (int i=0; i<no; ++i)
            {
                for (int j=0; j<no; ++j)
                {
                    for (int a=0; a<nv; ++a)
                    {
                        for (int b=0; b<nv; ++b)
                        {
                            Vs[i*n3+j*n2+a*n1+b] += B(i,a+no)*B(j,b+no);
                        }
                    }
                }
            }
        }

        Emp2=0.0;
        for (int i=0; i<no; ++i)
        {
            for (int j=0; j<no; ++j)
            {
                for (int a=0; a<nv; ++a)
                {
                    for (int b=0; b<nv; ++b)
                    {
                        T[i*n3+j*n2+a*n1+b] = (2*Vs[i*n3+j*n2+a*n1+b] - Vs[i*n3+j*n2+b*n1+a])/(d2(a+no,i)+d2(b+no,j));
                        Emp2 += real(T[i*n3+j*n2+a*n1+b]*Vs[i*n3+j*n2+a*n1+b]);
                    }
                }
            }
        }
        cout << std::setprecision(9) << "Mp2 Correlation energy: " << Emp2 << endl;
        
        cx_mat Poo(no,no); Poo.zeros();
        cx_mat Pvv(nv,nv); Pvv.zeros();
        cx_mat Pov(no,no); Pov.zeros();
        cx_mat Pmp2(RhoHF);
        
        if (params["Mp2NOGuess"])
        {
            // Now make the PDMs.
            // I'm taking these from JCC Ishimura, Pulay, and Nagase
            // and CPL 166,3 Frisch, HG, Pople.
            for (int i=0; i<no; ++i)
            {
                for (int j=0; j<no; ++j)
                {
                    for (int k=0; k<no; ++k)
                    {
                        for (int a=0; a<nv; ++a)
                        {
                            for (int b=0; b<nv; ++b)
                            {
                                Poo(i,j) -= 2.0*T[i*n3+k*n2+a*n1+b]*Vs[j*n3+k*n2+a*n1+b]/(d2(a+no,k)+d2(b+no,j));
                            }
                        }
                    }
                }
            }
            
            for (int i=0; i<no; ++i)
            {
                for (int j=0; j<no; ++j)
                {
                    for (int c=0; c<nv; ++c)
                    {
                        for (int a=0; a<nv; ++a)
                        {
                            for (int b=0; b<nv; ++b)
                            {
                                Pvv(a,b) += 2.0*T[i*n3+j*n2+a*n1+c]*Vs[i*n3+j*n2+b*n1+c]/(d2(b+no,i)+d2(c+no,j));
                            }
                        }
                    }
                }
            }
            cout << "Tr(oo) " << trace(Poo) << " Tr(vv)" << trace(Pvv) << endl;
            
            // finally do the lag. pieces.
            // Which require new integrals. (-_-)
            cx_mat Lai(nv,no); Lai.zeros();
            for(int R=0; R<n_aux; ++R)
            {
                cx_mat B = V.t()*BpqR.slice(R)*V;
                for (int i=0; i<no; ++i)
                {
                    for (int j=0; j<no; ++j)
                    {
                        for (int a=0; a<nv; ++a)
                        {
                            for (int b=0; b<nv; ++b)
                            {
                                for (int k=0; k<no; ++k)
                                {
                                    Lai(a,i) += Poo(j,k)*(4.0*B(i,a+no)*B(j,k)-2.0*B(i,k)*B(a+no,j));
                                    Lai(a,i) -= 4.0*T[j*n3+k*n2+a*n1+b]*B(i,j)*B(k,b+no);
                                }
                                for (int c=0; c<nv; ++c)
                                {
                                    Lai(a,i) += Pvv(b,c)*(4.0*B(i,a+no)*B(b+no,c+no)-2.0*B(i,b+no)*B(a+no,c+no));
                                    Lai(a,i) += 4.0*T[i*n3+j*n2+b*n1+c]*B(a+no,b+no)*B(j,c+no);
                                }
                            }
                        }
                    }
                }
            }
            Lai.print("Lai");
            // Solve the z-vector equation by iteration.
            // The equation has the form:
            // DP + L + VP = 0
            // Initalize P = L/D
            // Then calculate residual R = -L - VP
            // Then P = -R/D. Iterate.
            double dP=1.0; int iter=100;
            cx_mat dvo=d2.submat(no,0,n-1,no-1);
            cx_mat Pbj=-1.0*Lai/dvo;
            //        dvo.print("dvo");
            //        Pbj.print("Pbj0");
            cx_mat R(Pbj); R.zeros();
            while(dP>pow(10.0,-8.0) && iter>0)
            {
                // Calculate R.
                R = -1.0*Lai;
                for(int S=0; S<n_aux; ++S)
                {
                    cx_mat B = V.t()*BpqR.slice(S)*V;
                    for (int i=0; i<no; ++i)
                    {
                        for (int j=0; j<no; ++j)
                        {
                            for (int a=0; a<nv; ++a)
                            {
                                for (int b=0; b<nv; ++b)
                                {
                                    R(a,i) -= ( 4.0*B(a+no,i)*B(b+no,j)-B(a+no,j)*B(b+no,i)-B(a+no,b+no)*B(i,j) )*Pbj(b,j);
                                }
                            }
                        }
                    }
                }
                dP = accu(abs(Pbj-R/dvo));
                Pbj = R/dvo;
                cout << "Z-Iter:" << 100-iter << " dP: " << dP << endl;
                //            Pbj.print("Pbj");
                iter--;
            }
            Poo.print("Poo");
            Pvv.print("Pvv");
            Pmp2.submat(0,0,no-1,no-1) += Poo;
            Pmp2.submat(no,no,n-1,n-1) += Pvv;
            Pmp2.submat(0,no,no-1,n-1) += Pbj.t();
            Pmp2.submat(no,0,n-1,no-1) += Pbj;
            Pmp2.print("Pmp2");
        }
        return Pmp2;
    }
    
	// Just to see if this is faster than Job 32...
    // Rho is passed in the fock eigenbasis.
    // F is built in the lowdin basis,
    // then transformed into the fock eigenbasis and diagonalized.
	cx_mat FockBuild(cx_mat& Rho, cx_mat& V_, bool debug=false)
	{
        if (!B_Built)
        {
            cout << "cannot build fock without B" << endl;
            return Rho;
        }
        mat Xt = X.t();
        cx_mat Vt = V.t();
        cx_mat F(Rho); F.zeros();
        mat J(Rho.n_rows,Rho.n_cols); J.zeros();
        cx_mat K(Rho); K.zeros();
        cx_mat Rhol = V*Rho*Vt; // Get Rho into the lowdin basis for the fock build.
        mat h = 2.0*(Xt)*(H)*X; // h is stored in the ao basis.
        F.set_real(h);

#if HASUDR
#pragma omp declare reduction( + : cx_mat : \
std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< std::complex<double> >( )) ) initializer (omp_priv(omp_orig))

#pragma omp declare reduction( + : mat : \
std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< double >( )) ) initializer (omp_priv(omp_orig))
#endif
        
        cx_mat Rhot=Rhol.t();

#if HASUDR
#pragma omp parallel for reduction(+:J,K) schedule(static)
#endif
        for (int R=0; R<n_aux; ++R)
        {
            mat B=BpqR.slice(R);
            mat I1=real(Rhol%B); // Because Rho is hermitian and B is hermitian only the real part of this matrix should actually contribute to J
            J += 2.0*B*accu(I1); // J should be a Hermitian matrix.
            K -= B*(Rhot)*B;
        }
        
        F = 0.5*h + J + K;
        F = Vt*F*V;    // Move it back into the Prev eigenbasis.
        cx_mat Jx = Vt*J*V;    // Move it back into the Prev eigenbasis.
        cx_mat hx = Vt*h*V;    // Move it back into the Prev eigenbasis.
        K = Vt*K*V;    // Move it back into the Prev eigenbasis.

        F = 0.5*(F+F.t());
        
        double Eone= real(trace(Rho*hx));
        double EJ= real(trace(Rho*Jx));
        double EK= real(trace(Rho*K));
        Ehf = Eone + EJ + EK + nuclear_energy;

        cx_mat Vprime; eigs.zeros();
        if (!eig_sym(eigs, Vprime, F))
        {
            cout << "eig_sym failed to diagonalize fock matrix ... " << endl;
            throw 1;
        }
        
        if (params["Print"]>0.0)
        {
            cout << std::setprecision(9) << endl;
            cout << "HF Energy" << Ehf << endl;
            cout << "J Energy" << EJ << endl;
            cout << "K Energy" << EK << endl;
            cout << "nuclear_energy " << nuclear_energy << endl << endl;
            eigs.st().print("Eigs");
        }
        
        V_ = V = V*Vprime;
        C = X*V;
        return Vprime;
	}
	
	cx_mat Delta2(bool Sign=true)
	{
		cx_mat tore(eigs.n_elem,eigs.n_elem);
		for(int i=0; i<eigs.n_elem; ++i)
			for(int j=0; j<eigs.n_elem; ++j)
				tore(i,j) = (Sign)? (eigs(i) - eigs(j)) : eigs(i) + eigs(j);
        tore.diag().zeros();
		return tore;
	}
	
	// Updates BpqR
	void BuildB()
	{
        FileMan(FM_WRITE, FILE_MO_COEFS, FM_DP, n_mo*n_mo, 0, FM_BEG, X.memptr(), 0); // B is now built in the Lowdin basis and kept that way.
        BpqR*=0.0;
        mat Btmp(n_mo,n_aux);
        INTEGER ip[2], iq[2], iFile;
        iq[0] = 1; iq[1] = n_mo;
        ip[0] = 1; ip[1] = n_mo;
        int p_len = ip[1]-ip[0]+1;
        int q_len = iq[1]-iq[0]+1;
        iFile = FILE_3C2Ea_INTS;
        threading_policy::enable_omp_only();
        
        double* qas=qalloc_start();
        int LenV = megtot();
        formpqr(&iFile, iq, ip, qas, &LenV);
        threading_policy::pop();
        
        // Effn' qchem and its BS disk communication...
        // ($*%(*$%*@(@(*
        int fileNum = FILE_BIJQa_INTS;
        for (int q=0; q<q_len; q++)
        {
            int Off8 = q * p_len * n_aux; Btmp *= 0.0;
            LongFileMan(FM_READ, FILE_3C2Ea_INTS, FM_DP, p_len*n_aux, Off8, FM_BEG, Btmp.memptr(), 0);
            // Q.subcube( first_row, first_col, first_slice, last_row, last_col, last_slice )
            BpqR.subcube(q,0,0,q,n_mo-1,n_aux-1) += Btmp*RSinv;
        }
        // Ensure B is perfectly Hermitian.
        for (int R=0; R<n_aux; ++R)
            BpqR.slice(R) = 0.5*(BpqR.slice(R)+BpqR.slice(R).t());
        if (false)
        {
            // This works! Keeping it in as a check for now.
            double* f_diag = new double[n_mo];
            FileMan(FM_READ, FILE_MO_COEFS, FM_DP, n_mo, n_mo*n_mo*2 , FM_BEG, f_diag, 0);
            std::complex<double> ECorr = 0.0;
            cx_mat Vt = V.t();
            for (int R=0; R<n_aux; ++R)
            {
                cx_mat bR = Vt*BpqR.slice(R)*V;
                for (int S=0; S<n_aux; ++S)
                {
                    cx_mat bS = Vt*BpqR.slice(S)*V;
                    for (int i=0; i<n_occ; ++i)
                        for (int j=0; j<n_occ; ++j)
                            for (int a=n_occ; a<n_mo; ++a)
                                for (int b=n_occ; b<n_mo; ++b)
                                {
                                    double d = (f_diag[a]+f_diag[b]-f_diag[i]-f_diag[j]);
                                    {
                                        ECorr += bR(i,a)*bR(j,b)*(2.0*bS(i,a)*bS(j,b) - bS(i,b)*bS(j,a))/d;
                                    }
                                }
                }
            }
            cout << "Mp2 Correlation Energy: " << std::setprecision(9) << ECorr << endl;
        }
        B_Built = true;
	}
    
    bool cx_mat_equals(const arma::cx_mat& X,const arma::cx_mat& Y, double tol=pow(10.0,-12.0)) const
    {
        if (X.n_rows != Y.n_rows)
            return false;
        if (X.n_cols != Y.n_cols)
            return false;
        bool close(false);
        if(arma::max(arma::max(arma::abs(X-Y))) < tol)
            close = true;
        return close;
    }
    
    void UpdateB(const cx_mat& V_)
    {
        if (cx_mat_equals(V_,VCur))
            return;
        int n2=n_mo*n_mo;
        cx_mat Vi(n2,n2); Vi.zeros();
#if HASUDR
#pragma omp declare reduction( + : cx_mat : \
std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< std::complex<double> >( )) ) initializer (omp_priv(omp_orig))
        
#pragma omp parallel for reduction(+: Vi) schedule(static)
#endif
        for(int R=0; R<n_aux; ++R)
        {
            cx_mat B = V_.t()*BpqR.slice(R)*V_; // No need to xform 5 times/iteration. I could do this once for a 5x speedup.
            Vi += kron(B,B).st();
        }
        IntCur = Vi;
        VCur = V_;
    }
    
    void Split_RK4_Step_MMUT(const arma::vec& eigval, const arma::cx_mat& Cu , const arma::cx_mat& oldrho, arma::cx_mat& newrho, const double tnow, const double dt)
    {
        cx_mat Ud = exp(eigval*0.5*j*dt);
        cx_mat U = Cu*diagmat(Ud)*Cu.t();
        cx_mat RhoHalfStepped = U*oldrho*U.t();
        
        arma::cx_mat k1(newrho),k2(newrho),k3(newrho),k4(newrho),v2(newrho),v3(newrho),v4(newrho);
        k1.zeros(); k2.zeros(); k3.zeros(); k4.zeros();
        v2.zeros(); v3.zeros(); v4.zeros();
        
        RhoDot( RhoHalfStepped, k1,tnow);
        v2 = (dt/2.0) * k1;
        v2 += RhoHalfStepped;
        RhoDot(  RhoHalfStepped, k2,tnow+(dt/2.0));
        v3 = (dt/2.0) * k2;
        v3 += RhoHalfStepped;
        RhoDot(  RhoHalfStepped, k3,tnow+(dt/2.0));
        v4 = (dt) * k3;
        v4 += RhoHalfStepped;
        RhoDot(  RhoHalfStepped, k4,tnow+dt);
        newrho = RhoHalfStepped;
        newrho += dt*(1.0/6.0)*k1;
        newrho += dt*(2.0/6.0)*k2;
        newrho += dt*(2.0/6.0)*k3;
        newrho += dt*(1.0/6.0)*k4;
        newrho = U*newrho*U.t();
    }
    
    
    // Two mmut steps of length dt/2 and one ee2 step (calc'd with RK4)
    double step(rhf* the_scf, FieldMatrices* Mus, arma::vec& eigs_, arma::cx_mat& V_,  arma::cx_mat& Rho_, arma::cx_mat& RhoM12_, const double tnow , const double dt,bool& IsOn)
    {
        cx_mat newrho(Rho_);
        cx_mat Rot = FockBuild(Rho_, V_); // updates eigs, C, V, F, etc. etc. etc.
        UpdateB(V_); // Updates CurrInts to avoid rebuilds in the steps that follow.
        
        vec noc0; cx_mat nos0;
        if (params["Print"]>0.0)
            eig_sym(noc0,nos0,Rho_);
        
        eigs_ = eigs;
        Mus->update(X*V_);
        Rho_ = Rot.t()*Rho_*Rot;
        RhoM12_ = Rot.t()*RhoM12_*Rot;
        
        // Make the exponential propagator.
        arma::cx_mat F(Rho_); F.zeros(); F.set_real(diagmat(eigs));
        Mus->ApplyField(F,tnow,IsOn);
        vec Gd; cx_mat Cu;
        eig_sym(Gd,Cu,F);
        
        // Full step RhoM12 to make new RhoM12.
        cx_mat NewRhoM12(Rho_);
        cx_mat NewRho(Rho_);
        Split_RK4_Step_MMUT(Gd, Cu, RhoM12_, NewRhoM12, tnow, dt);
        
        // Half step that to make the new Rho.
        Split_RK4_Step_MMUT(Gd, Cu, NewRhoM12, NewRho, tnow, dt/2.0);
        Rho_ = 0.5*(NewRho+NewRho.t());//0.5*(NewRho+NewRho.t());
        RhoM12_ = 0.5*(NewRhoM12+NewRhoM12.t());//0.5*(NewRhoM12+NewRhoM12.t());
        
        //Epseudo=real(RhoDot(Rho_, NewRho, tnow));
        
        // Get the change in noccs.
        if (params["Print"]>1.0)
        {
            vec noc1; cx_mat nos1;
            eig_sym(noc1,nos1,Rho_);
            noc1-=noc0;
            noc1.st().print("Delta N-Occ Nums: ");
        }
        return Ehf;
    }
    
    cx RhoDot(const cx_mat& Rho_, cx_mat& RhoDot_, const double& time)
    {
        int n=n_mo;
        int n2=n_mo*n_mo;
        int n3=n2*n_mo;
        int n4=n3*n_mo;
        
        cx_mat Rho(Rho_);
        for (int i=0; i<n; ++i)
        {
            if (real(Rho(i,i))<0)
                Rho(i,i)=0.0;
            if( abs(Rho(i,i))>1.0)
                Rho(i,i)=1.0;
        }
#if HASUDR
#pragma omp declare reduction( + : cx_mat : \
std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< std::complex<double> >( )) ) initializer (omp_priv(omp_orig))
#endif
        
        if (params["RIFock"]!=0.0)
            return 0.0;
        
        threading_policy::enable_omp_only();
        
        cx_mat Eta(Rho); Eta.eye();
        Eta -= Rho;
        
        cx_vec rho(Rho.diag());
        cx_vec eta(Eta.diag());
        
        cx_mat Out(RhoDot_); Out.zeros();
        cx_mat d2 = Delta2();
        
        cx_mat Vsm(n2,n2); Vsm.zeros();
        cx_mat Vtm(n2,n2); Vtm.zeros();
        cx* Vs = Vsm.memptr();
        cx* Vt = Vtm.memptr();
        
        cx* Vi = IntCur.memptr();
        
#pragma omp parallel for schedule(static)
        for(int p=0; p<n_mo; ++p)
        {
            for(int q=0; q<n_mo; ++q)
            {
                for(int r=0; r<n_mo; ++r)
                {
                    for(int s=0; s<n_mo; ++s)
                    {
                        // This appears to work great.
                        cx d = (d2(p,r)+d2(q,s));
                        
                        Vt[p*n3+q*n2+r*n+s] = (Vi[p*n3+q*n2+r*n+s]);
                        
                        if(  (abs(d)>pow(10.0,-10.0)) )
                            Vs[p*n3+q*n2+r*n+s] = (1.0/(j*d))*(2.0*Vi[p*n3+q*n2+r*n+s] - Vi[p*n3+q*n2+s*n+r]);
                    }
                }
            }
        }
        
        cx_mat I1(n,n), I2(n,n); I1.zeros(); I2.zeros();
        cx_mat I3(n,n), I4(n,n); I3.zeros(); I4.zeros();
        {
            //#pragma omp parallel for reduction(+:I1,I2,I3,I4,Out) schedule(static)
            for(int x=0; x<n_mo; ++x)
            {
                int y=x;
                //for(int y=0; y<n_mo; ++y)
                {
                    for(int s=0; s<n_mo; ++s)
                    {
                        for(int u=0; u<n_mo; ++u)
                        {
                            for(int w=0; w<n_mo; ++w)
                            {
                                // Precise changes I would like to make to this:
                                // I would like the ia ia MP2 like terms.
                                // More precise spin integration.
                                
                                
                                // These are ESC like.
                                I1(y,x) +=  (0.5)*Vs[n3*s+n2*y+n*u+w]*Vt[n3*u+n2*w+n*s+x]*rho(u)*rho(w)*eta(s); // Dsign. v (y) o  o
                                I1(y,x) += (-0.5)*Vs[n3*s+n2*u+n*w+x]*Vt[n3*w+n2*y+n*s+u]*rho(w)*eta(s)*eta(u); // Dsign. v  v  o (x)

                                I2(x,y) +=  (0.5)*Vs[n3*u+n2*w+n*s+y]*Vt[n3*s+n2*x+n*u+w]*rho(u)*rho(w)*eta(s); // Dsign. o  o  v (y)
                                I2(x,y) += (-0.5)*Vs[n3*w+n2*x+n*s+u]*Vt[n3*s+n2*u+n*w+y]*rho(w)*eta(s)*eta(u); // Dsign. o (x) v  v
                                
                            }
                        }
                    }
                }
            }
            
            for(int x=0; x<n_mo; ++x)
            {
                int y=x;
               // for(int y=0; y<n_mo; ++y)
                {
                    for(int s=0; s<n_mo; ++s)
                    {
                        for(int u=0; u<n_mo; ++u)
                        {
                            for(int w=0; w<n_mo; ++w)
                            {
                                
                                // These are Mp2 like terms, I'm not sure a diagonal appx. is good here....
                                // would make the method O(N^5)
                                // The correct secular approximation fixes the outer indices of Rho Rho and eta eta to be the same.
                                I3(y,x) += (-0.5)*Vs[n3*s+n2*u+n*w+x]*Vt[n3*w+n2*y+n*s+u]*rho(w)*eta(s)*eta(u); // Dsign v v   o (x)
                                I3(y,x) += (-0.5)*Vs[n3*w+n2*y+n*s+u]*Vt[n3*s+n2*u+n*w+x]*rho(w)*eta(s)*eta(u); // Dsign o (y) v  v
                                I4(y,x) +=  (0.5)*Vs[n3*s+n2*y+n*u+w]*Vt[n3*u+n2*w+n*s+x]*rho(u)*rho(w)*eta(s); // Dsign v (y) o o
                                I4(y,x) +=  (0.5)*Vs[n3*u+n2*w+n*s+x]*Vt[n3*s+n2*y+n*u+w]*rho(u)*rho(w)*eta(s); // Dsign o o v  (x)
                                
                            }
                        }
                    }
                }
            }
            
            I3.print("I3");
            I4.print("I4");
            
            cx_mat t1 = Rho*I1*Eta;
            cx_mat t2 = Eta*I2*Rho;
            t1.print("t1");
            t2.print("t2");
            cx_mat t3 = Rho*I3*Rho;
            t3.print("t3");
            cx_mat t4 = Eta*I4*Eta;
            t3.print("t4");
            
            Out += Rho*I1*Eta;
            Out += Eta*I2*Rho; // Mp2 Like Terms.
            Out += Rho*I3*Rho; //vo|ov
            Out += Eta*I4*Eta; // Ex.State Correlation Terms.
            
        }
        
        RhoDot_.zeros();
        RhoDot_ += lambda*0.5*(Out+Out.t());
        if (params["Print"]>=1.0)
        {
            RhoDot_.print("RhoDot(fast)");
            //        Rho.print("Rho");
            //        cout << "d(trace)" << trace(RhoDot_) << endl;
            
            // Check positivity.
            if (params["Print"]>1.0)
            {
                bool Positive=true;
                for (int i=0; i<n_mo; ++i)
                {
                    if(real(Rho(i,i)+0.02*RhoDot_(i,i))<0 or real(Rho(i,i)+0.02*RhoDot_(i,i))>1)
                    {
                        cout << " Nonpositive " << i  << Rho(i,i) << RhoDot_(i,i) << Rho(i,i)+0.02*RhoDot_(i,i) << endl;
                        Positive = false;
                    }
                }
            }
            
        }
        threading_policy::pop();
        
        return (trace(RhoDot_%Rho));
    }
    
    cx RhoDotGood(const cx_mat& Rho_, cx_mat& RhoDot_, const double& time)
    {
        int n=n_mo;
        int n2=n_mo*n_mo;
        int n3=n2*n_mo;
        int n4=n3*n_mo;
        
        cx_mat Rho(Rho_);
        for (int i=0; i<n; ++i)
        {
            if (real(Rho(i,i))<0)
                Rho(i,i)=0.0;
            if( abs(Rho(i,i))>1.0)
                Rho(i,i)=1.0;
        }
#if HASUDR
#pragma omp declare reduction( + : cx_mat : \
std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< std::complex<double> >( )) ) initializer (omp_priv(omp_orig))
#endif
        
        if (params["RIFock"]!=0.0)
            return 0.0;
        
        threading_policy::enable_omp_only();
        
        cx_mat Eta(Rho); Eta.eye();
        Eta -= Rho;
        
        cx_vec rho(Rho.diag());
        cx_vec eta(Eta.diag());
        
        cx_mat Out(RhoDot_); Out.zeros();
        cx_mat d2 = Delta2();
        
        cx_mat Vsm(n2,n2); Vsm.zeros();
        cx_mat Vtm(n2,n2); Vtm.zeros();
        cx* Vs = Vsm.memptr();
        cx* Vt = Vtm.memptr();
        
        cx* Vi = IntCur.memptr();
        
#pragma omp parallel for schedule(static)
        for(int p=0; p<n_mo; ++p)
        {
            for(int q=0; q<n_mo; ++q)
            {
                for(int r=0; r<n_mo; ++r)
                {
                    for(int s=0; s<n_mo; ++s)
                    {
                        // This appears to work great.
                        cx d = (d2(p,r)+d2(q,s));
                        
                        Vt[p*n3+q*n2+r*n+s] = (2.0*Vi[p*n3+q*n2+r*n+s] - Vi[p*n3+q*n2+s*n+r]);
                        
                        if(  (abs(d)>pow(10.0,-10.0)) )
                            Vs[p*n3+q*n2+r*n+s] = (1.0/(-j*d))*(2.0*Vi[p*n3+q*n2+r*n+s] - Vi[p*n3+q*n2+s*n+r]);
                    }
                }
            }
        }
        
        cx_mat I1(n,n), I2(n,n); I1.zeros(); I2.zeros();
        cx_mat I3(n,n), I4(n,n); I3.zeros(); I4.zeros();
        {
//#pragma omp parallel for reduction(+:I1,I2,I3,I4,Out) schedule(static)
            for(int x=0; x<n_mo; ++x)
            {
                //for(int y=0; y<n_mo; ++y)
                int y=x;
                {
                    for(int s=0; s<n_mo; ++s)
                    {
                        for(int u=0; u<n_mo; ++u)
                        {
                            for(int w=0; w<n_mo; ++w)
                            {
                                
                                // All these contributions should be positive since etaia=-rhoia These are Mp2 like terms.
                                I1(y,x) +=  (0.5)*Vs[n3*s+n2*y+n*u+w]*Vt[n3*u+n2*w+n*s+x]*rho(u)*rho(w)*eta(s); // Dsign. v (y) o  o
                                I1(y,x) += (-0.5)*Vs[n3*s+n2*u+n*w+x]*Vt[n3*w+n2*y+n*s+u]*rho(w)*eta(s)*eta(u); // Dsign. v  v  o (x)
                                I2(x,y) +=  (0.5)*Vs[n3*u+n2*w+n*s+y]*Vt[n3*s+n2*x+n*u+w]*rho(u)*rho(w)*eta(s); // Dsign. o  o  v (y)
                                I2(x,y) += (-0.5)*Vs[n3*w+n2*x+n*s+u]*Vt[n3*s+n2*u+n*w+y]*rho(w)*eta(s)*eta(u); // Dsign. o (x) v  v

                                // These are the excited state correlation terms (should be negative)
                                I3(y,x) += (-0.5)*Vs[n3*s+n2*u+n*w+x]*Vt[n3*w+n2*y+n*s+u]*rho(w)*eta(s)*eta(u); // Dsign v v   o (x)
                                I3(y,x) += (-0.5)*Vs[n3*w+n2*y+n*s+u]*Vt[n3*s+n2*u+n*w+x]*rho(w)*eta(s)*eta(u); // Dsign o (y) v  v
                                I4(y,x) +=  (0.5)*Vs[n3*s+n2*y+n*u+w]*Vt[n3*u+n2*w+n*s+x]*rho(u)*rho(w)*eta(s);
                                I4(y,x) +=  (0.5)*Vs[n3*u+n2*w+n*s+x]*Vt[n3*s+n2*y+n*u+w]*rho(u)*rho(w)*eta(s);

                            }
                        }
                    }
                }
            }
            
            cx_mat t1 = Rho*I1*Eta;
            cx_mat t2 = Eta*I2*Rho; // Mp2 Like Terms.
            t1.print("t1");
            t2.print("t2");
            cx_mat t3 = Rho*I3*Rho;
            cx_mat t4 = Eta*I4*Eta;
            t3.print("t3");
            t4.print("t4");
            
            Out -= Rho*I1*Eta;
            Out -= Eta*I2*Rho; // Mp2 Like Terms.
            Out -= Rho*I3*Rho; //ov|vo
            Out -= Eta*I4*Eta; // Ex.State Correlation Terms.
            
        }

        RhoDot_.zeros();
        RhoDot_ += lambda*0.5*(Out+Out.t());
        if (params["Print"]>=1.0)
        {
            RhoDot_.print("RhoDot(fast)");
            //        Rho.print("Rho");
            //        cout << "d(trace)" << trace(RhoDot_) << endl;
            
            // Check positivity.
            if (params["Print"]>1.0)
            {
                bool Positive=true;
                for (int i=0; i<n_mo; ++i)
                {
                    if(real(Rho(i,i)+0.02*RhoDot_(i,i))<0 or real(Rho(i,i)+0.02*RhoDot_(i,i))>1)
                    {
                        cout << " Nonpositive " << i  << Rho(i,i) << RhoDot_(i,i) << Rho(i,i)+0.02*RhoDot_(i,i) << endl;
                        Positive = false;
                    }
                }
            }
            
        }
        threading_policy::pop();
        
        return (trace(RhoDot_%Rho));
    }
    
    
    // Double secular term from
    // the Yanai version of the expectation value.
    cx RhoDotYanai(const cx_mat& Rho_, cx_mat& RhoDot_, const double& time)
    {
        int n=n_mo;
        int n2=n_mo*n_mo;
        int n3=n2*n_mo;
        int n4=n3*n_mo;
        
        cx_mat Rho(Rho_);
        for (int i=0; i<n; ++i)
        {
            if (real(Rho(i,i))<0)
                Rho(i,i)=0.0;
            if( abs(Rho(i,i))>1.0)
                Rho(i,i)=1.0;
        }
#if HASUDR
#pragma omp declare reduction( + : cx_mat : \
std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< std::complex<double> >( )) ) initializer (omp_priv(omp_orig))
#endif
        
        if (params["RIFock"]!=0.0)
            return 0.0;
        
        threading_policy::enable_omp_only();
        cx_mat Eta(Rho); Eta.eye();
        Eta -= Rho;
        cx_mat Out(RhoDot_); Out.zeros();
        cx_mat d2 = Delta2();
        
        cx_mat Vs_ssm(n2,n2); Vs_ssm.zeros();
        cx_mat Vs_osm(n2,n2); Vs_osm.zeros();
        
        cx_mat Vsd_ssm(n2,n2); Vsd_ssm.zeros();
        cx_mat Vsd_osm(n2,n2); Vsd_osm.zeros();
        
        cx_mat Vt_ssm(n2,n2); Vt_ssm.zeros();
        cx_mat Vt_osm(n2,n2); Vt_osm.zeros();
        
        cx* Vs_ss = Vs_ssm.memptr();
        cx* Vs_os = Vs_osm.memptr();
        
        cx* Vsd_ss = Vsd_ssm.memptr();
        cx* Vsd_os = Vsd_osm.memptr();
        
        cx* Vt_ss = Vt_ssm.memptr();
        cx* Vt_os = Vt_osm.memptr();
        
        cx* Vi = IntCur.memptr();
        // The integrals (IntCur) have been transformed and updated by UpdateB following the fock build.
        //        cx* Vi = IntCur.memptr(); // Note DO NOT OVERWRITE Vi!!!!!!!
        
#pragma omp parallel for schedule(static)
        for(int p=0; p<n_mo; ++p)
        {
            for(int q=0; q<n_mo; ++q)
            {
                for(int r=0; r<n_mo; ++r)
                {
                    for(int s=0; s<n_mo; ++s)
                    {
                        // This appears to work great.
                        cx d = (d2(p,r)+d2(q,s));
                        
                        Vt_ss[p*n3+q*n2+r*n+s] = (Vi[p*n3+q*n2+r*n+s] - Vi[p*n3+q*n2+s*n+r]);
                        Vt_os[p*n3+q*n2+r*n+s] = (Vi[p*n3+q*n2+r*n+s]);
                        
                        cx ep(0.00001,0.0);
                        if(  (abs(d)>pow(10.0,-10.0)) )
                        {
                            Vsd_ss[p*n3+q*n2+r*n+s] = (1.0/(j*d))*(Vi[p*n3+q*n2+r*n+s] - Vi[p*n3+q*n2+s*n+r]);
                            Vsd_os[p*n3+q*n2+r*n+s] = (1.0/(j*d))*(Vi[p*n3+q*n2+r*n+s]);
                            
                            Vs_ss[p*n3+q*n2+r*n+s] = (1.0/(j*d))*(Vi[p*n3+q*n2+r*n+s] - Vi[p*n3+q*n2+s*n+r]);
                            Vs_os[p*n3+q*n2+r*n+s] = (1.0/(j*d))*(Vi[p*n3+q*n2+r*n+s]);
                        }
                        /*
                         if( ! (abs(real(d))<pow(10.0,-10.0) ))
                         {
                         Vs_ss[p*n3+q*n2+r*n+s] = (j/(d))*(Vi[p*n3+q*n2+r*n+s] - Vi[p*n3+q*n2+s*n+r]);
                         Vs_os[p*n3+q*n2+r*n+s] = (j/(d))*(Vi[p*n3+q*n2+r*n+s]);
                         }
                         */
                        
                    }
                }
            }
        }
        
        cx_mat I1(n,n); I1.zeros();
        cx_mat I2(n,n); I2.zeros();
        cx_mat I3(n,n); I3.zeros();
        cx_mat I4(n,n); I4.zeros();
        cx_mat OutDiag(Out); OutDiag.zeros();
        
        
        // Includes both p, q terms
        {
            // q intermediate terms.
#if HASUDR
#pragma omp parallel for reduction(+:I1,Out) schedule(static)
#endif
            for(int q=0; q<n_mo; ++q)
            {
                for(int r=0; r<n_mo; ++r)
                {
                    for(int s=0; s<n_mo; ++s)
                    {
                        for(int t=0; t<n_mo; ++t)
                        {
                            I1(q,q)+=(-1.)*Vs_os[n3*t+n2*s+n*q+r]*Vt_os[n3*q+n2*r+n*t+s]*Rho(r,r);
                            I1(q,q)+=(-1.)*Vs_os[n3*t+n2*s+n*q+r]*Vt_os[n3*q+n2*r+n*t+s]*Rho(r,r);
                            I1(q,q)+=(-1.)*Vs_ss[n3*t+n2*s+n*q+r]*Vt_ss[n3*q+n2*r+n*t+s]*Rho(r,r);
                            I1(q,r)+=(1.)*Vs_os[n3*r+n2*s+n*q+t]*Vt_os[n3*q+n2*t+n*r+s]*Rho(s,s)*Rho(r,q);
                            I1(q,r)+=(1.)*Vs_ss[n3*s+n2*r+n*q+t]*Vt_ss[n3*q+n2*t+n*s+r]*Rho(s,s)*Rho(r,q);
                            I1(q,r)+=(-1.)*Vs_os[n3*q+n2*s+n*q+t]*Vt_os[n3*r+n2*t+n*r+s]*Rho(s,s)*Rho(r,q);
                            I1(q,r)+=(1.)*Vs_ss[n3*q+n2*s+n*q+t]*Vt_ss[n3*r+n2*t+n*s+r]*Rho(s,s)*Rho(r,q);
                            I1(q,r)+=(-2.)*Vs_os[n3*r+n2*t+n*q+s]*Vt_os[n3*q+n2*s+n*r+t]*Rho(s,s)*Rho(r,q);
                            I1(q,r)+=(-2.)*Vs_ss[n3*r+n2*t+n*q+s]*Vt_ss[n3*q+n2*s+n*r+t]*Rho(s,s)*Rho(r,q);
                            I1(q,r)+=(2.)*Vs_os[n3*q+n2*t+n*q+s]*Vt_os[n3*r+n2*s+n*r+t]*Rho(s,s)*Rho(r,q);
                            I1(q,r)+=(-2.)*Vs_ss[n3*q+n2*t+n*q+s]*Vt_ss[n3*s+n2*r+n*r+t]*Rho(s,s)*Rho(r,q);
                            I1(q,q)+=(1.)*Vs_os[n3*q+n2*r+n*q+t]*Vt_ss[n3*s+n2*t+n*r+s]*Rho(r,s)*Rho(s,r);
                            I1(q,q)+=(1.)*Vs_ss[n3*q+n2*r+n*q+t]*Vt_ss[n3*s+n2*t+n*r+s]*Rho(r,s)*Rho(s,r);
                            I1(q,q)+=(-2.)*Vs_os[n3*t+n2*s+n*q+r]*Vt_os[n3*q+n2*r+n*t+s]*Rho(r,s)*Rho(s,r);
                            I1(q,q)+=(-2.)*Vs_ss[n3*s+n2*t+n*q+r]*Vt_ss[n3*q+n2*r+n*s+t]*Rho(r,s)*Rho(s,r);
                            I1(q,q)+=(1.)*Vs_ss[n3*q+n2*s+n*q+t]*Vt_os[n3*t+n2*r+n*s+r]*Rho(s,s)*Rho(r,r);
                            I1(q,q)+=(-1.)*Vs_os[n3*q+n2*s+n*q+t]*Vt_os[n3*r+n2*t+n*s+r]*Rho(s,s)*Rho(r,r);
                            I1(q,q)+=(1.)*Vs_os[n3*q+n2*s+n*q+t]*Vt_ss[n3*t+n2*r+n*s+r]*Rho(s,s)*Rho(r,r);
                            I1(q,q)+=(-1.)*Vs_ss[n3*q+n2*s+n*q+t]*Vt_ss[n3*r+n2*t+n*s+r]*Rho(s,s)*Rho(r,r);
                            I1(q,q)+=(2.)*Vs_os[n3*r+n2*t+n*q+s]*Vt_os[n3*q+n2*s+n*r+t]*Rho(s,s)*Rho(r,r);
                            I1(q,q)+=(2.)*Vs_os[n3*t+n2*r+n*q+s]*Vt_os[n3*q+n2*s+n*t+r]*Rho(s,s)*Rho(r,r);
                            I1(q,q)+=(2.)*Vs_ss[n3*r+n2*t+n*q+s]*Vt_ss[n3*q+n2*s+n*r+t]*Rho(s,s)*Rho(r,r);
                            I1(q,s)+=(-1.)*Vs_ss[n3*r+n2*s+n*q+t]*Vt_ss[n3*q+n2*t+n*r+s]*Rho(r,q)*Rho(s,r);
                            I1(q,s)+=(2.)*Vs_ss[n3*q+n2*t+n*q+r]*Vt_ss[n3*r+n2*s+n*s+t]*Rho(r,q)*Rho(s,r);
                            I1(q,q)+=(-1.)*Vs_os[n3*s+n2*r+n*q+t]*Vt_os[n3*q+n2*t+n*s+r]*Rho(s,s)*Rho(r,r);
                            I1(q,q)+=(-1.)*Vs_os[n3*r+n2*s+n*q+t]*Vt_os[n3*q+n2*t+n*r+s]*Rho(s,s)*Rho(r,r);
                            I1(q,q)+=(-1.)*Vs_ss[n3*s+n2*r+n*q+t]*Vt_ss[n3*q+n2*t+n*s+r]*Rho(s,s)*Rho(r,r);
                            I1(q,q)+=(2.)*Vs_ss[n3*q+n2*t+n*q+s]*Vt_os[n3*s+n2*r+n*r+t]*Rho(s,s)*Rho(r,r);
                            I1(q,q)+=(-2.)*Vs_os[n3*q+n2*t+n*q+s]*Vt_os[n3*r+n2*s+n*r+t]*Rho(s,s)*Rho(r,r);
                            I1(q,q)+=(-2.)*Vs_os[n3*q+n2*t+n*q+s]*Vt_ss[n3*r+n2*s+n*r+t]*Rho(s,s)*Rho(r,r);
                            I1(q,q)+=(2.)*Vs_ss[n3*q+n2*t+n*q+s]*Vt_ss[n3*s+n2*r+n*r+t]*Rho(s,s)*Rho(r,r);
                        }
                    }
                }
            }
#if HASUDR
#pragma omp parallel for reduction(+:I2,Out) schedule(static)
#endif
            for(int p=0; p<n_mo; ++p)
            {
                for(int r=0; r<n_mo; ++r)
                {
                    for(int s=0; s<n_mo; ++s)
                    {
                        for(int q=0; q<n_mo; ++q)
                        {
                            Out(p,q)+=(1.)*Vs_os[n3*p+n2*s+n*p+r]*Vt_os[n3*q+n2*r+n*q+s]*Rho(r,r)*Rho(p,q);
                            Out(p,q)+=(1.)*Vs_os[n3*q+n2*r+n*q+s]*Vt_os[n3*p+n2*s+n*p+r]*Rho(r,r)*Rho(p,q);
                            Out(p,q)+=(1.)*Vs_ss[n3*p+n2*s+n*p+r]*Vt_ss[n3*q+n2*r+n*q+s]*Rho(r,r)*Rho(p,q);
                            Out(p,q)+=(1.)*Vs_ss[n3*q+n2*r+n*q+s]*Vt_ss[n3*p+n2*s+n*p+r]*Rho(r,r)*Rho(p,q);
                            Out(p,q)+=(-1.)*Vs_os[n3*p+n2*r+n*p+r]*Vt_os[n3*q+n2*s+n*q+s]*Rho(r,s)*Rho(s,r)*Rho(p,q);
                            Out(p,q)+=(-1.)*Vs_os[n3*p+n2*r+n*p+s]*Vt_os[n3*q+n2*s+n*q+r]*Rho(s,s)*Rho(r,r)*Rho(p,q);
                            Out(p,q)+=(-1.)*Vs_os[n3*p+n2*s+n*p+r]*Vt_os[n3*p+n2*r+n*p+s]*Rho(p,p)*Rho(r,r)*Rho(p,q);
                            
                            Out(p,q)+=(-1.)*Vs_os[n3*p+n2*s+n*p+s]*Vt_ss[n3*q+n2*r+n*q+r]*Rho(s,s)*Rho(r,q)*Rho(p,r);
                            Out(p,q)+=(-1.)*Vs_os[n3*q+n2*r+n*q+r]*Vt_os[n3*p+n2*s+n*p+s]*Rho(r,s)*Rho(s,r)*Rho(p,q);
                            Out(p,q)+=(-1.)*Vs_os[n3*q+n2*r+n*q+s]*Vt_os[n3*p+n2*s+n*p+r]*Rho(s,s)*Rho(r,r)*Rho(p,q);
                            Out(p,q)+=(-1.)*Vs_os[n3*q+n2*s+n*q+s]*Vt_ss[n3*p+n2*r+n*p+r]*Rho(s,s)*Rho(r,q)*Rho(p,r);
                            Out(p,q)+=(-1.)*Vs_ss[n3*p+n2*r+n*p+r]*Vt_ss[n3*q+n2*s+n*q+s]*Rho(r,s)*Rho(s,r)*Rho(p,q);
                            Out(p,q)+=(-1.)*Vs_ss[n3*p+n2*r+n*p+s]*Vt_ss[n3*p+n2*s+n*p+r]*Rho(p,p)*Rho(r,r)*Rho(p,q);
                            Out(p,q)+=(-1.)*Vs_ss[n3*p+n2*r+n*p+s]*Vt_ss[n3*q+n2*s+n*q+r]*Rho(s,s)*Rho(r,r)*Rho(p,q);
                            Out(p,q)+=(-1.)*Vs_ss[n3*p+n2*r+n*p+s]*Vt_ss[n3*q+n2*s+n*q+r]*Rho(r,q)*Rho(s,r)*Rho(p,s);
                            Out(p,q)+=(-1.)*Vs_ss[n3*p+n2*s+n*p+r]*Vt_ss[n3*p+n2*r+n*p+s]*Rho(p,p)*Rho(r,r)*Rho(p,q);
                            
                            Out(p,q)+=(-1.)*Vs_ss[n3*p+n2*s+n*p+s]*Vt_ss[n3*q+n2*r+n*q+r]*Rho(s,s)*Rho(r,q)*Rho(p,r);
                            Out(p,q)+=(-1.)*Vs_ss[n3*q+n2*r+n*q+r]*Vt_ss[n3*p+n2*s+n*p+s]*Rho(r,s)*Rho(s,r)*Rho(p,q);
                            Out(p,q)+=(-1.)*Vs_ss[n3*q+n2*r+n*q+s]*Vt_ss[n3*p+n2*s+n*p+r]*Rho(s,s)*Rho(r,r)*Rho(p,q);
                            Out(p,q)+=(-1.)*Vs_ss[n3*q+n2*r+n*q+s]*Vt_ss[n3*p+n2*s+n*p+r]*Rho(r,q)*Rho(s,r)*Rho(p,s);
                            Out(p,q)+=(-1.)*Vs_ss[n3*q+n2*s+n*q+s]*Vt_ss[n3*p+n2*r+n*p+r]*Rho(s,s)*Rho(r,q)*Rho(p,r);
                            Out(p,q)+=(1.)*Vs_os[n3*q+n2*s+n*q+r]*Vt_os[n3*p+n2*r+n*p+s]*Rho(r,s)*Rho(s,r)*Rho(p,q);
                            Out(p,q)+=(1.)*Vs_os[n3*q+n2*s+n*q+s]*Vt_os[n3*p+n2*r+n*p+r]*Rho(s,s)*Rho(r,r)*Rho(p,q);
                            Out(p,q)+=(1.)*Vs_os[n3*q+n2*s+n*q+s]*Vt_ss[n3*p+n2*r+n*p+r]*Rho(s,s)*Rho(r,r)*Rho(p,q);
                            Out(p,q)+=(1.)*Vs_ss[n3*p+n2*r+n*p+s]*Vt_ss[n3*p+n2*s+n*p+r]*Rho(p,r)*Rho(r,p)*Rho(p,q);
                            Out(p,q)+=(1.)*Vs_ss[n3*p+n2*r+n*p+s]*Vt_ss[n3*q+n2*s+n*q+r]*Rho(s,s)*Rho(r,q)*Rho(p,r);
                            Out(p,q)+=(1.)*Vs_ss[n3*p+n2*s+n*p+s]*Vt_ss[n3*q+n2*r+n*q+r]*Rho(r,q)*Rho(s,r)*Rho(p,s);
                            Out(p,q)+=(1.)*Vs_ss[n3*q+n2*r+n*q+s]*Vt_ss[n3*q+n2*s+n*q+r]*Rho(q,r)*Rho(r,q)*Rho(p,q);
                            Out(p,q)+=(1.)*Vs_ss[n3*q+n2*s+n*q+r]*Vt_ss[n3*p+n2*r+n*p+s]*Rho(r,s)*Rho(s,r)*Rho(p,q);
                            Out(p,q)+=(1.)*Vs_ss[n3*q+n2*s+n*q+s]*Vt_os[n3*p+n2*r+n*p+r]*Rho(s,s)*Rho(r,r)*Rho(p,q);
                            Out(p,q)+=(1.)*Vs_ss[n3*q+n2*s+n*q+s]*Vt_ss[n3*p+n2*r+n*p+r]*Rho(s,s)*Rho(r,r)*Rho(p,q);
                            Out(p,q)+=(2.)*Vs_ss[n3*q+n2*s+n*q+r]*Vt_ss[n3*q+n2*r+n*q+s]*Rho(q,r)*Rho(r,q)*Rho(p,q);
                        }
                        
                        
                        for(int t=0; t<n_mo; ++t)
                        {
                            I2(p,p)+=(-1.)*Vs_os[n3*p+n2*r+n*t+s]*Vt_os[n3*t+n2*s+n*p+r]*Rho(r,r);
                            I2(p,p)+=(-1.)*Vs_os[n3*p+n2*r+n*t+s]*Vt_os[n3*t+n2*s+n*p+r]*Rho(r,r);
                            I2(p,p)+=(-1.)*Vs_ss[n3*p+n2*r+n*t+s]*Vt_ss[n3*t+n2*s+n*p+r]*Rho(r,r);
                            I2(p,r)+=(-1.)*Vs_os[n3*p+n2*s+n*r+t]*Vt_os[n3*r+n2*t+n*p+s]*Rho(p,r)*Rho(s,s);
                            I2(p,r)+=(-1.)*Vs_ss[n3*p+n2*s+n*r+t]*Vt_ss[n3*r+n2*t+n*p+s]*Rho(p,r)*Rho(s,s);
                            I2(p,r)+=(2.)*Vs_os[n3*p+n2*t+n*r+s]*Vt_os[n3*r+n2*s+n*p+t]*Rho(p,r)*Rho(s,s);
                            I2(p,r)+=(2.)*Vs_ss[n3*p+n2*t+n*s+r]*Vt_ss[n3*s+n2*r+n*p+t]*Rho(p,r)*Rho(s,s);
                            I2(p,r)+=(2.)*Vs_os[n3*p+n2*s+n*p+t]*Vt_os[n3*r+n2*t+n*r+s]*Rho(p,r)*Rho(s,s);
                            I2(p,r)+=(-2.)*Vs_ss[n3*p+n2*s+n*p+t]*Vt_ss[n3*r+n2*t+n*s+r]*Rho(p,r)*Rho(s,s);
                            I2(p,r)+=(-1.)*Vs_os[n3*p+n2*t+n*p+s]*Vt_os[n3*r+n2*s+n*r+t]*Rho(p,r)*Rho(s,s);
                            I2(p,r)+=(1.)*Vs_ss[n3*p+n2*t+n*p+s]*Vt_ss[n3*s+n2*r+n*r+t]*Rho(p,r)*Rho(s,s);
                            I2(p,r)+=(1.)*Vs_ss[n3*p+n2*s+n*r+t]*Vt_ss[n3*r+n2*t+n*p+s]*Rho(p,s)*Rho(s,r);
                            I2(p,r)+=(-2.)*Vs_ss[n3*p+n2*s+n*p+t]*Vt_ss[n3*r+n2*t+n*r+s]*Rho(p,s)*Rho(s,r);
                            I2(p,p)+=(-2.)*Vs_os[n3*p+n2*s+n*t+r]*Vt_os[n3*t+n2*r+n*p+s]*Rho(r,s)*Rho(s,r);
                            I2(p,p)+=(-2.)*Vs_ss[n3*p+n2*s+n*r+t]*Vt_ss[n3*r+n2*t+n*p+s]*Rho(r,s)*Rho(s,r);
                            I2(p,p)+=(-1.)*Vs_os[n3*p+n2*r+n*p+t]*Vt_ss[n3*s+n2*t+n*r+s]*Rho(r,s)*Rho(s,r);
                            I2(p,p)+=(-1.)*Vs_ss[n3*p+n2*r+n*p+t]*Vt_ss[n3*s+n2*t+n*r+s]*Rho(r,s)*Rho(s,r);
                            I2(p,p)+=(2.)*Vs_os[n3*p+n2*r+n*s+t]*Vt_os[n3*s+n2*t+n*p+r]*Rho(s,s)*Rho(r,r);
                            I2(p,p)+=(2.)*Vs_os[n3*p+n2*r+n*t+s]*Vt_os[n3*t+n2*s+n*p+r]*Rho(s,s)*Rho(r,r);
                            I2(p,p)+=(2.)*Vs_ss[n3*p+n2*r+n*s+t]*Vt_ss[n3*s+n2*t+n*p+r]*Rho(s,s)*Rho(r,r);
                            I2(p,p)+=(-1.)*Vs_os[n3*p+n2*t+n*s+r]*Vt_os[n3*s+n2*r+n*p+t]*Rho(s,s)*Rho(r,r);
                            I2(p,p)+=(-1.)*Vs_os[n3*p+n2*t+n*r+s]*Vt_os[n3*r+n2*s+n*p+t]*Rho(s,s)*Rho(r,r);
                            I2(p,p)+=(-1.)*Vs_ss[n3*p+n2*t+n*s+r]*Vt_ss[n3*s+n2*r+n*p+t]*Rho(s,s)*Rho(r,r);
                            I2(p,p)+=(-1.)*Vs_ss[n3*p+n2*s+n*p+t]*Vt_os[n3*t+n2*r+n*s+r]*Rho(s,s)*Rho(r,r);
                            I2(p,p)+=(1.)*Vs_os[n3*p+n2*s+n*p+t]*Vt_os[n3*r+n2*t+n*s+r]*Rho(s,s)*Rho(r,r);
                            I2(p,p)+=(-1.)*Vs_os[n3*p+n2*s+n*p+t]*Vt_ss[n3*t+n2*r+n*s+r]*Rho(s,s)*Rho(r,r);
                            I2(p,p)+=(1.)*Vs_ss[n3*p+n2*s+n*p+t]*Vt_ss[n3*r+n2*t+n*s+r]*Rho(s,s)*Rho(r,r);
                            I2(p,p)+=(-2.)*Vs_ss[n3*p+n2*t+n*p+s]*Vt_os[n3*s+n2*r+n*r+t]*Rho(s,s)*Rho(r,r);
                            I2(p,p)+=(2.)*Vs_os[n3*p+n2*t+n*p+s]*Vt_os[n3*r+n2*s+n*r+t]*Rho(s,s)*Rho(r,r);
                            I2(p,p)+=(2.)*Vs_os[n3*p+n2*t+n*p+s]*Vt_ss[n3*r+n2*s+n*r+t]*Rho(s,s)*Rho(r,r);
                            I2(p,p)+=(-2.)*Vs_ss[n3*p+n2*t+n*p+s]*Vt_ss[n3*s+n2*r+n*r+t]*Rho(s,s)*Rho(r,r);
                            
                            //{{Sign, 1}, {{Vs_os, 2}, {p, t}, {r, s}}, {{Vt_os, 2}, {r, s}, {p, t}}, {{\[Gamma], 1}, {s}, {s}}, {{\[Gamma], 1}, {r}, {r}}}
                            Out(p,p)+=(1.)*Vs_os[n3*p+n2*t+n*r+s]*Vt_os[n3*r+n2*s+n*p+t]*Rho(s,s)*Rho(r,r);
                            //{{Sign, 1}, {{Vs_os, 2}, {p, t}, {s, r}}, {{Vt_os, 2}, {s, r}, {p, t}}, {{\[Gamma], 1}, {s}, {s}}, {{\[Gamma], 1}, {r}, {r}}}
                            Out(p,p)+=(1.)*Vs_os[n3*p+n2*t+n*s+r]*Vt_os[n3*s+n2*r+n*p+t]*Rho(s,s)*Rho(r,r);
                            //{{Sign, 1}, {{Vs_os, 2}, {r, s}, {p, t}}, {{Vt_os, 2}, {p, t}, {r, s}}, {{\[Gamma], 1}, {s}, {s}}, {{\[Gamma], 1}, {r}, {r}}}
                            Out(p,p)+=(1.)*Vs_os[n3*r+n2*s+n*p+t]*Vt_os[n3*p+n2*t+n*r+s]*Rho(s,s)*Rho(r,r);
                            //{{Sign, 1}, {{Vs_os, 2}, {s, r}, {p, t}}, {{Vt_os, 2}, {p, t}, {s, r}}, {{\[Gamma], 1}, {s}, {s}}, {{\[Gamma], 1}, {r}, {r}}}
                            Out(p,p)+=(1.)*Vs_os[n3*s+n2*r+n*p+t]*Vt_os[n3*p+n2*t+n*s+r]*Rho(s,s)*Rho(r,r);
                            //{{Sign, 1}, {{Vs_ss, 2}, {p, t}, {s, r}}, {{Vt_ss, 2}, {s, r}, {p, t}}, {{\[Gamma], 1}, {s}, {s}}, {{\[Gamma], 1}, {r}, {r}}}
                            Out(p,p)+=(1.)*Vs_ss[n3*p+n2*t+n*s+r]*Vt_ss[n3*s+n2*r+n*p+t]*Rho(s,s)*Rho(r,r);
                            //{{Sign, 1}, {{Vs_ss, 2}, {s, r}, {p, t}}, {{Vt_ss, 2}, {p, t}, {s, r}}, {{\[Gamma], 1}, {s}, {s}}, {{\[Gamma], 1}, {r}, {r}}}
                            Out(p,p)+=(1.)*Vs_ss[n3*s+n2*r+n*p+t]*Vt_ss[n3*p+n2*t+n*s+r]*Rho(s,s)*Rho(r,r);
                            //{{Sign, -1}, {{Vs_os, 2}, {p, r}, {s, t}}, {{Vt_os, 2}, {s, t}, {p, r}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Gamma], 1}, {s}, {s}}, {{\[Gamma], 1}, {r}, {r}}}
                            Out(p,p)+=(-1.)*Vs_os[n3*p+n2*r+n*s+t]*Vt_os[n3*s+n2*t+n*p+r]*Rho(t,t)*Rho(s,s)*Rho(r,r);
                            //{{Sign, -1}, {{Vs_os, 2}, {p, r}, {t, r}}, {{Vt_os, 2}, {t, s}, {p, s}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Gamma], 1}, {r}, {s}}, {{\[Gamma], 1}, {s}, {r}}}
                            Out(p,p)+=(-1.)*Vs_os[n3*p+n2*r+n*t+r]*Vt_os[n3*t+n2*s+n*p+s]*Rho(t,t)*Rho(r,s)*Rho(s,r);
                            //{{Sign, -1}, {{Vs_os, 2}, {p, r}, {t, s}}, {{Vt_os, 2}, {t, s}, {p, r}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Gamma], 1}, {s}, {s}}, {{\[Gamma], 1}, {r}, {r}}}
                            Out(p,p)+=(-1.)*Vs_os[n3*p+n2*r+n*t+s]*Vt_os[n3*t+n2*s+n*p+r]*Rho(t,t)*Rho(s,s)*Rho(r,r);
                            //{{Sign, -1}, {{Vs_os, 2}, {p, t}, {r, t}}, {{Vt_ss, 2}, {r, s}, {p, s}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Gamma], 1}, {r}, {s}}, {{\[Gamma], 1}, {s}, {r}}}
                            Out(p,p)+=(-1.)*Vs_os[n3*p+n2*t+n*r+t]*Vt_ss[n3*r+n2*s+n*p+s]*Rho(t,t)*Rho(r,s)*Rho(s,r);
                            //{{Sign, -1}, {{Vs_os, 2}, {r, s}, {p, s}}, {{Vt_os, 2}, {p, t}, {r, t}}, {{\[Gamma], 1}, {s}, {t}}, {{\[Gamma], 1}, {t}, {s}}, {{\[Gamma], 1}, {r}, {r}}}
                            Out(p,p)+=(-1.)*Vs_os[n3*r+n2*s+n*p+s]*Vt_os[n3*p+n2*t+n*r+t]*Rho(s,t)*Rho(t,s)*Rho(r,r);
                            //{{Sign, -1}, {{Vs_os, 2}, {r, s}, {p, t}}, {{Vt_os, 2}, {p, t}, {r, s}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Gamma], 1}, {s}, {s}}, {{\[Gamma], 1}, {r}, {r}}}
                            Out(p,p)+=(-1.)*Vs_os[n3*r+n2*s+n*p+t]*Vt_os[n3*p+n2*t+n*r+s]*Rho(t,t)*Rho(s,s)*Rho(r,r);
                            //{{Sign, -1}, {{Vs_os, 2}, {r, t}, {p, t}}, {{Vt_ss, 2}, {p, s}, {s, r}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Gamma], 1}, {s}, {s}}, {{\[Gamma], 1}, {r}, {r}}}
                            Out(p,p)+=(-1.)*Vs_os[n3*r+n2*t+n*p+t]*Vt_ss[n3*p+n2*s+n*s+r]*Rho(t,t)*Rho(s,s)*Rho(r,r);
                            //{{Sign, -1}, {{Vs_os, 2}, {s, r}, {p, t}}, {{Vt_os, 2}, {p, t}, {s, r}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Gamma], 1}, {s}, {s}}, {{\[Gamma], 1}, {r}, {r}}}
                            Out(p,p)+=(-1.)*Vs_os[n3*s+n2*r+n*p+t]*Vt_os[n3*p+n2*t+n*s+r]*Rho(t,t)*Rho(s,s)*Rho(r,r);
                            //{{Sign, -1}, {{Vs_ss, 2}, {p, r}, {t, r}}, {{Vt_ss, 2}, {t, s}, {p, s}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Gamma], 1}, {r}, {s}}, {{\[Gamma], 1}, {s}, {r}}}
                            Out(p,p)+=(-1.)*Vs_ss[n3*p+n2*r+n*t+r]*Vt_ss[n3*t+n2*s+n*p+s]*Rho(t,t)*Rho(r,s)*Rho(s,r);
                            //{{Sign, -1}, {{Vs_ss, 2}, {p, r}, {t, s}}, {{Vt_ss, 2}, {t, s}, {p, r}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Gamma], 1}, {s}, {s}}, {{\[Gamma], 1}, {r}, {r}}}
                            Out(p,p)+=(-1.)*Vs_ss[n3*p+n2*r+n*t+s]*Vt_ss[n3*t+n2*s+n*p+r]*Rho(t,t)*Rho(s,s)*Rho(r,r);
                            //{{Sign, -1}, {{Vs_ss, 2}, {p, t}, {r, s}}, {{Vt_ss, 2}, {r, s}, {p, t}}, {{\[Gamma], 1}, {t}, {s}}, {{\[Gamma], 1}, {r}, {t}}, {{\[Gamma], 1}, {s}, {r}}}
                            Out(p,p)+=(-1.)*Vs_ss[n3*p+n2*t+n*r+s]*Vt_ss[n3*r+n2*s+n*p+t]*Rho(t,s)*Rho(r,t)*Rho(s,r);
                            //{{Sign, -1}, {{Vs_ss, 2}, {s, r}, {p, s}}, {{Vt_ss, 2}, {p, t}, {t, r}}, {{\[Gamma], 1}, {s}, {t}}, {{\[Gamma], 1}, {t}, {s}}, {{\[Gamma], 1}, {r}, {r}}}
                            Out(p,p)+=(-1.)*Vs_ss[n3*s+n2*r+n*p+s]*Vt_ss[n3*p+n2*t+n*t+r]*Rho(s,t)*Rho(t,s)*Rho(r,r);
                            //{{Sign, -1}, {{Vs_ss, 2}, {s, r}, {p, t}}, {{Vt_ss, 2}, {p, t}, {s, r}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Gamma], 1}, {s}, {s}}, {{\[Gamma], 1}, {r}, {r}}}
                            Out(p,p)+=(-1.)*Vs_ss[n3*s+n2*r+n*p+t]*Vt_ss[n3*p+n2*t+n*s+r]*Rho(t,t)*Rho(s,s)*Rho(r,r);
                            //{{Sign, -1}, {{Vs_ss, 2}, {t, r}, {p, s}}, {{Vt_ss, 2}, {p, s}, {t, r}}, {{\[Gamma], 1}, {t}, {s}}, {{\[Gamma], 1}, {r}, {t}}, {{\[Gamma], 1}, {s}, {r}}}
                            Out(p,p)+=(-1.)*Vs_ss[n3*t+n2*r+n*p+s]*Vt_ss[n3*p+n2*s+n*t+r]*Rho(t,s)*Rho(r,t)*Rho(s,r);
                            //{{Sign, -1}, {{Vs_ss, 2}, {t, r}, {p, t}}, {{Vt_os, 2}, {p, s}, {r, s}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Gamma], 1}, {s}, {s}}, {{\[Gamma], 1}, {r}, {r}}}
                            Out(p,p)+=(-1.)*Vs_ss[n3*t+n2*r+n*p+t]*Vt_os[n3*p+n2*s+n*r+s]*Rho(t,t)*Rho(s,s)*Rho(r,r);
                            //{{Sign, -1}, {{Vs_ss, 2}, {t, s}, {p, t}}, {{Vt_ss, 2}, {p, r}, {r, s}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Gamma], 1}, {r}, {s}}, {{\[Gamma], 1}, {s}, {r}}}
                            Out(p,p)+=(-1.)*Vs_ss[n3*t+n2*s+n*p+t]*Vt_ss[n3*p+n2*r+n*r+s]*Rho(t,t)*Rho(r,s)*Rho(s,r);
                            //{{Sign, 1}, {{Vs_os, 2}, {p, s}, {t, r}}, {{Vt_os, 2}, {t, r}, {p, s}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Gamma], 1}, {r}, {s}}, {{\[Gamma], 1}, {s}, {r}}}
                            Out(p,p)+=(1.)*Vs_os[n3*p+n2*s+n*t+r]*Vt_os[n3*t+n2*r+n*p+s]*Rho(t,t)*Rho(r,s)*Rho(s,r);
                            //{{Sign, 1}, {{Vs_os, 2}, {r, t}, {p, s}}, {{Vt_os, 2}, {p, s}, {r, t}}, {{\[Gamma], 1}, {s}, {t}}, {{\[Gamma], 1}, {t}, {s}}, {{\[Gamma], 1}, {r}, {r}}}
                            Out(p,p)+=(1.)*Vs_os[n3*r+n2*t+n*p+s]*Vt_os[n3*p+n2*s+n*r+t]*Rho(s,t)*Rho(t,s)*Rho(r,r);
                            //{{Sign, 1}, {{Vs_os, 2}, {r, t}, {p, t}}, {{Vt_os, 2}, {p, s}, {r, s}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Gamma], 1}, {s}, {s}}, {{\[Gamma], 1}, {r}, {r}}}
                            Out(p,p)+=(1.)*Vs_os[n3*r+n2*t+n*p+t]*Vt_os[n3*p+n2*s+n*r+s]*Rho(t,t)*Rho(s,s)*Rho(r,r);
                            //{{Sign, 1}, {{Vs_os, 2}, {s, t}, {p, t}}, {{Vt_ss, 2}, {p, r}, {r, s}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Gamma], 1}, {r}, {s}}, {{\[Gamma], 1}, {s}, {r}}}
                            Out(p,p)+=(1.)*Vs_os[n3*s+n2*t+n*p+t]*Vt_ss[n3*p+n2*r+n*r+s]*Rho(t,t)*Rho(r,s)*Rho(s,r);
                            //{{Sign, 1}, {{Vs_ss, 2}, {p, r}, {r, s}}, {{Vt_ss, 2}, {t, s}, {p, t}}, {{\[Gamma], 1}, {t}, {s}}, {{\[Gamma], 1}, {r}, {t}}, {{\[Gamma], 1}, {s}, {r}}}
                            Out(p,p)+=(1.)*Vs_ss[n3*p+n2*r+n*r+s]*Vt_ss[n3*t+n2*s+n*p+t]*Rho(t,s)*Rho(r,t)*Rho(s,r);
                            //{{Sign, 1}, {{Vs_ss, 2}, {p, s}, {t, r}}, {{Vt_ss, 2}, {t, r}, {p, s}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Gamma], 1}, {r}, {s}}, {{\[Gamma], 1}, {s}, {r}}}
                            Out(p,p)+=(1.)*Vs_ss[n3*p+n2*s+n*t+r]*Vt_ss[n3*t+n2*r+n*p+s]*Rho(t,t)*Rho(r,s)*Rho(s,r);
                            //{{Sign, 1}, {{Vs_ss, 2}, {p, t}, {t, r}}, {{Vt_ss, 2}, {r, s}, {p, s}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Gamma], 1}, {r}, {s}}, {{\[Gamma], 1}, {s}, {r}}}
                            Out(p,p)+=(1.)*Vs_ss[n3*p+n2*t+n*t+r]*Vt_ss[n3*r+n2*s+n*p+s]*Rho(t,t)*Rho(r,s)*Rho(s,r);
                            //{{Sign, 1}, {{Vs_ss, 2}, {t, r}, {p, s}}, {{Vt_ss, 2}, {p, s}, {t, r}}, {{\[Gamma], 1}, {s}, {t}}, {{\[Gamma], 1}, {t}, {s}}, {{\[Gamma], 1}, {r}, {r}}}
                            Out(p,p)+=(1.)*Vs_ss[n3*t+n2*r+n*p+s]*Vt_ss[n3*p+n2*s+n*t+r]*Rho(s,t)*Rho(t,s)*Rho(r,r);
                            //{{Sign, 1}, {{Vs_ss, 2}, {t, r}, {p, t}}, {{Vt_ss, 2}, {p, s}, {s, r}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Gamma], 1}, {s}, {s}}, {{\[Gamma], 1}, {r}, {r}}}
                            Out(p,p)+=(1.)*Vs_ss[n3*t+n2*r+n*p+t]*Vt_ss[n3*p+n2*s+n*s+r]*Rho(t,t)*Rho(s,s)*Rho(r,r);
                            
                        } // Loop t
                        
                        
                        // rs pp terms.
                        Out(p,p)+=(-1.)*Vs_os[n3*p+n2*r+n*p+s]*Vt_os[n3*p+n2*s+n*p+r]*Rho(r,r)*Rho(p,p);
                        Out(p,p)+=(-1.)*Vs_os[n3*p+n2*s+n*p+r]*Vt_os[n3*p+n2*r+n*p+s]*Rho(r,r)*Rho(p,p);
                        Out(p,p)+=(-1.)*Vs_ss[n3*p+n2*r+n*p+s]*Vt_ss[n3*p+n2*s+n*p+r]*Rho(r,r)*Rho(p,p);
                        Out(p,p)+=(-1.)*Vs_ss[n3*p+n2*s+n*p+r]*Vt_ss[n3*p+n2*r+n*p+s]*Rho(r,r)*Rho(p,p);
                        Out(p,p)+=(1.)*Vs_os[n3*p+n2*s+n*r+s]*Vt_os[n3*r+n2*s+n*p+s]*Rho(s,s)*Rho(r,r);
                        Out(p,p)+=(1.)*Vs_ss[n3*p+n2*s+n*s+r]*Vt_ss[n3*s+n2*r+n*p+s]*Rho(s,s)*Rho(r,r);
                        Out(p,p)+=(-1.)*Vs_os[n3*p+n2*s+n*p+r]*Vt_os[n3*p+n2*r+n*p+s]*Rho(r,s)*Rho(s,r)*Rho(p,p);
                        Out(p,p)+=(-1.)*Vs_os[n3*p+n2*s+n*p+s]*Vt_os[n3*p+n2*r+n*p+r]*Rho(s,s)*Rho(r,r)*Rho(p,p);
                        Out(p,p)+=(-1.)*Vs_os[n3*p+n2*s+n*p+s]*Vt_ss[n3*p+n2*r+n*p+r]*Rho(s,s)*Rho(r,r)*Rho(p,p);
                        Out(p,p)+=(-1.)*Vs_ss[n3*p+n2*r+n*p+s]*Vt_ss[n3*p+n2*s+n*p+r]*Rho(s,s)*Rho(r,p)*Rho(p,r);
                        Out(p,p)+=(-1.)*Vs_ss[n3*p+n2*s+n*p+r]*Vt_ss[n3*p+n2*r+n*p+s]*Rho(r,s)*Rho(s,r)*Rho(p,p);
                        Out(p,p)+=(-1.)*Vs_ss[n3*p+n2*s+n*p+s]*Vt_os[n3*p+n2*r+n*p+r]*Rho(s,s)*Rho(r,r)*Rho(p,p);
                        Out(p,p)+=(-1.)*Vs_ss[n3*p+n2*s+n*p+s]*Vt_ss[n3*p+n2*r+n*p+r]*Rho(s,s)*Rho(r,r)*Rho(p,p);
                        Out(p,p)+=(-1.)*Vs_ss[n3*p+n2*s+n*p+s]*Vt_ss[n3*p+n2*r+n*p+r]*Rho(r,p)*Rho(s,r)*Rho(p,s);
                        Out(p,p)+=(1.)*Vs_ss[n3*r+n2*s+n*p+r]*Vt_ss[n3*p+n2*r+n*r+s]*Rho(r,r)*Rho(r,s)*Rho(s,r);
                        Out(p,p)+=(2.)*Vs_os[n3*p+n2*r+n*p+r]*Vt_os[n3*p+n2*s+n*p+s]*Rho(r,s)*Rho(s,r)*Rho(p,p);
                        Out(p,p)+=(2.)*Vs_os[n3*p+n2*r+n*p+s]*Vt_os[n3*p+n2*s+n*p+r]*Rho(s,s)*Rho(r,r)*Rho(p,p);
                        Out(p,p)+=(2.)*Vs_os[n3*p+n2*s+n*p+s]*Vt_ss[n3*p+n2*r+n*p+r]*Rho(s,s)*Rho(r,p)*Rho(p,r);
                        Out(p,p)+=(2.)*Vs_ss[n3*p+n2*r+n*p+r]*Vt_ss[n3*p+n2*s+n*p+s]*Rho(r,s)*Rho(s,r)*Rho(p,p);
                        Out(p,p)+=(2.)*Vs_ss[n3*p+n2*r+n*p+s]*Vt_ss[n3*p+n2*s+n*p+r]*Rho(s,s)*Rho(r,r)*Rho(p,p);
                        Out(p,p)+=(2.)*Vs_ss[n3*p+n2*r+n*p+s]*Vt_ss[n3*p+n2*s+n*p+r]*Rho(r,p)*Rho(s,r)*Rho(p,s);
                        Out(p,p)+=(2.)*Vs_ss[n3*p+n2*s+n*p+s]*Vt_ss[n3*p+n2*r+n*p+r]*Rho(s,s)*Rho(r,p)*Rho(p,r);
                        
                        
                    } //loop s
                    
                    
                    Out(p,p)+=(-1.)*Vs_os[n3*p+n2*r+n*p+r]*Vt_os[n3*p+n2*r+n*p+r]*Rho(r,r)*Rho(p,p);
                    Out(p,p)+=(-1.)*Vs_ss[n3*p+n2*r+n*p+r]*Vt_ss[n3*p+n2*r+n*p+r]*Rho(r,r)*Rho(p,p);
                    Out(p,p)+=(-1.)*Vs_ss[n3*p+n2*r+n*p+r]*Vt_ss[n3*p+n2*r+n*p+r]*Rho(r,p)*Rho(r,r)*Rho(p,r);
                    
                    for (int q=0; q<n_mo; ++q)
                    {
                        Out(p,q)+=(1.)*Vs_os[n3*p+n2*r+n*p+r]*Vt_os[n3*q+n2*r+n*q+r]*Rho(r,r)*Rho(p,q);
                        Out(p,q)+=(1.)*Vs_ss[n3*p+n2*r+n*p+r]*Vt_ss[n3*q+n2*r+n*q+r]*Rho(r,r)*Rho(p,q);
                        Out(p,q)+=(1.)*Vs_ss[n3*q+n2*r+n*q+r]*Vt_ss[n3*p+n2*r+n*p+r]*Rho(r,q)*Rho(r,r)*Rho(p,r);
                    }
                }
            }
            
            Out += Rho*I1.st(); I1.zeros();
            Out += I2*Rho; I2.zeros();
        }
        
        RhoDot_.zeros();
        RhoDot_ += lambda*0.5*(Out+Out.t());
        if (params["Print"]>=1.0)
        {
            RhoDot_.print("RhoDot(fast)");
            //        Rho.print("Rho");
            //        cout << "d(trace)" << trace(RhoDot_) << endl;
            
            // Check positivity.
            if (params["Print"]>1.0)
            {
                bool Positive=true;
                for (int i=0; i<n_mo; ++i)
                {
                    if(real(Rho(i,i)+0.02*RhoDot_(i,i))<0 or real(Rho(i,i)+0.02*RhoDot_(i,i))>1)
                    {
                        cout << " Nonpositive " << i  << Rho(i,i) << RhoDot_(i,i) << Rho(i,i)+0.02*RhoDot_(i,i) << endl;
                        Positive = false;
                    }
                }
            }
            
        }
        threading_policy::pop();
        
        return (trace(RhoDot_%Rho));
    }
    
// ********************
// ********************
    
    // Most up-to-date secular term.
    void RhoDotSec(const cx_mat& Rho_, cx_mat& RhoDot_, const double& time)
    {
        // User defined reductions for stupid OMP.
#if HASUDR
#pragma omp declare reduction( + : cx_mat : \
std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< std::complex<double> >( )) ) initializer (omp_priv(omp_orig))
        
#pragma omp declare reduction( + : std::vector<cx> : std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< std::complex<double> >( )) ) initializer (omp_priv(omp_orig))
#endif
        if (params["RIFock"]!=0.0)
            return;
        
        int n=n_mo;
        int n2=n_mo*n_mo;
        int n3=n2*n_mo;
        int n4=n3*n_mo;
        
        // Cannot allow negative populations as input.
        // Small numerical integration errors can allow these to percolate through the EOM
        cx_mat Rho=Rho_;
        for (int i=0; i<n; ++i)
        {
            if (real(Rho(i,i))<0)
                Rho(i,i)=0.0;
            if( abs(Rho(i,i))>1.0)
                Rho(i,i)=1.0;
        }
        
        // To Debug.
        //        RhoDotSecFull(Rho,  RhoDot_, time);
        //        RhoDot_.zeros();
        
        threading_policy::enable_omp_only();
        cx_mat Eta(Rho); Eta.eye();
        Eta -= Rho;
        cx_mat Out(RhoDot_); Out.zeros();
        
        cx_mat d2 = Delta2();
        
        // figure out how to kron these suckas.
        cx_mat Vsm(n2,n2); Vsm.zeros();
        cx_mat V13_1m(n2,n2); V13_1m.zeros();
        cx_mat V14_1m(n2,n2); V14_1m.zeros();
        cx_mat V24_2m(n2,n2); V24_2m.zeros();
        cx_mat V23_2m(n2,n2); V23_2m.zeros();
        
#if HASUDR
#pragma omp parallel for reduction(+:Vsm, V13_1m, V14_1m, V23_2m, V24_2m) schedule(static)
#endif
        for(int R=0; R<n_aux; ++R)
        {
            cx_mat Osc = exp(j*time*d2);
            cx_mat B = V.t()*BpqR.slice(R)*V;
            cx_mat Bo = B;//%Osc;
            cx_mat Bg = Bo*Rho;
            cx_mat eB = Eta*Bo;
            cx_mat eBg = eB*Rho;
            
            Vsm += kron(B,B).st();
            V13_1m += kron(eBg,Bo).st();
            V14_1m += kron(eB,Bg).st();
            V23_2m += kron(Bg,eB).st();
            V24_2m += kron(Bo,eBg).st();
        }
        
        cx* Vs = Vsm.memptr();
        cx* V13_1 = V13_1m.memptr();
        cx* V14_1 = V14_1m.memptr();
        cx* V23_2 = V23_2m.memptr();
        cx* V24_2 = V24_2m.memptr();
        
        #pragma omp parallel for schedule(static)
        for(int p=0; p<n_mo; ++p)
        {
            for(int q=0; q<n_mo; ++q)
            {
                for(int r=0; r<n_mo; ++r)
                {
                    for(int s=0; s<n_mo; ++s)
                    {
//                        std::complex<double> ep(0.01, 0.0);
//                        std::complex<double> d = 1.0/(j*(d2(p,r)+d2(q,s))+ep);
//                        std::complex<double> d = (ep)/(j*(d2(p,r)+d2(q,s))+ep);
                        cx d=(d2(p,r)+d2(q,s));
                        if( abs(real(d))<pow(10.0,-14.0) )
                            d = 0.0;
                        else
                            d = 1.0/(j*d);
                        Vs[p*n3+q*n2+r*n+s] *= d;
                    }
                }
            }
        }
        
        std::vector<cx> I1(n2),I2(n2),I3(n2),I4(n2),I5(n2),I6(n2),I7(n2);
        std::fill(I1.begin(), I1.end(), std::complex<double>(0.0,0.0));
        std::fill(I2.begin(), I2.end(), std::complex<double>(0.0,0.0));
        std::fill(I3.begin(), I3.end(), std::complex<double>(0.0,0.0));
        std::fill(I4.begin(), I4.end(), std::complex<double>(0.0,0.0));
        std::fill(I5.begin(), I5.end(), std::complex<double>(0.0,0.0));
        std::fill(I6.begin(), I6.end(), std::complex<double>(0.0,0.0));
        std::fill(I7.begin(), I7.end(), std::complex<double>(0.0,0.0));
        
#if HASUDR
#pragma omp parallel for reduction(+:I1,I2,I3,I4,I5,I6,I7) schedule(static)
#endif
        for(int r=0; r<n_mo; ++r)
        {
            for(int s=0; s<n_mo; ++s)
            {
                for(int p=0; p<n_mo; ++p)
                {
                    for(int t=0; t<n_mo; ++t) // Having t as the last index will reduce cache misses.
                    {
                        I1[p*n+r] += ((Vs[n3*r+n2*s+n*t+p]-Vs[n3*r+n2*s+n*p+t])*V23_2[n3*p+n2*t+n*s+r] - (Vs[n3*r+n2*s+n*t+p]-2.0*(Vs[n3*r+n2*s+n*p+t]))*V24_2[n3*p+n2*t+n*r+s]);
                        I2[p*n+r] += Vs[n3*r+n2*s+n*p+t]*V24_2[n3*r+n2*t+n*p+s];
                        I3[p*n+s] += (Vs[n3*r+n2*s+n*p+t]-Vs[n3*r+n2*s+n*t+p])*(V14_1[n3*t+n2*s+n*p+r]-V13_1[n3*t+n2*s+n*r+p]);
                        I4[p*n+s] += Vs[n3*r+n2*s+n*p+t]*V23_2[n3*p+n2*t+n*r+s];
                    }
                }
                for(int q=0; q<n_mo; ++q)
                {
                    for(int t=0; t<n_mo; ++t)
                    {
                        
                        I5[q*n+s] += Vs[n3*q+n2*r+n*s+t]*(V14_1[n3*t+n2*s+n*q+r]-V13_1[n3*t+n2*s+n*r+q]-V24_2[n3*s+n2*t+n*q+r]);
                        I6[q*n+t] += ((Vs[n3*q+n2*r+n*s+t]-2.0*Vs[n3*q+n2*r+n*t+s])*V24_2[n3*q+n2*s+n*t+r]-(Vs[n3*q+n2*r+n*s+t]-Vs[n3*q+n2*r+n*t+s])*V23_2[n3*q+n2*s+n*r+t]);
                        I7[q*n+t] += Vs[n3*q+n2*r+n*s+t]*(V13_1[n3*s+n2*t+n*r+q]-2.0*V14_1[n3*s+n2*t+n*q+r]);
                    }
                }
            }
        }
        
#if HASUDR
#pragma omp parallel for reduction(+:Out) schedule(static)
#endif
        for(int p=0; p<n_mo; ++p)
        {
            for(int q=0; q<n_mo; ++q)
            {
                cx Outpq(0.0,0.0);
                for(int r=0; r<n_mo; ++r)
                {
                    Outpq+=I1[p*n+r]*Rho(r,r)*Eta(p,q);
                    Outpq+=I2[p*n+r]*Rho(r,p)*Eta(r,q);
                    Outpq+=I3[p*n+r]*Rho(r,p)*Eta(r,q);
                    Outpq+=I4[p*n+r]*Rho(r,r)*Eta(p,q);
                    
                    Outpq+=I5[q*n+r]*Rho(p,q)*Eta(r,r);
                    Outpq+=I6[q*n+r]*Rho(p,r)*Eta(q,r);
                    Outpq+=I7[q*n+r]*Rho(p,q)*Eta(r,r);
                    
                    for(int s=0; s<n_mo; ++s)
                    {
                        //61{{Sign, -1}, {{Vs, 2}, {p, u}, {p, r}}, {{Vt, 2}, {p, s}, {p, t}}, {{\[Gamma], 1}, {p}, {p}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                        Outpq+=(-2.)*Vs[n3*p+n2*r+n*p+s]*V24_2[n3*p+n2*s+n*p+r]*Rho(p,p)*Eta(p,q);
                        //61{{Sign, 1}, {{Vs, 2}, {p, u}, {p, r}}, {{Vt, 2}, {p, s}, {t, p}}, {{\[Gamma], 1}, {p}, {p}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                        Outpq+=(1.)*Vs[n3*p+n2*r+n*p+s]*V23_2[n3*p+n2*s+n*r+p]*Rho(p,p)*Eta(p,q);
                        //61{{Sign, 1}, {{Vs, 2}, {p, u}, {r, p}}, {{Vt, 2}, {p, s}, {p, t}}, {{\[Gamma], 1}, {p}, {p}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                        Outpq+=(1.)*Vs[n3*p+n2*r+n*s+p]*V24_2[n3*p+n2*s+n*p+r]*Rho(p,p)*Eta(p,q);
                        //61{{Sign, -1}, {{Vs, 2}, {p, u}, {r, p}}, {{Vt, 2}, {p, s}, {t, p}}, {{\[Gamma], 1}, {p}, {p}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                        Outpq+=(-1.)*Vs[n3*p+n2*r+n*s+p]*V23_2[n3*p+n2*s+n*r+p]*Rho(p,p)*Eta(p,q);
                        
                        //62{{Sign, 1}, {{Vs, 2}, {q, u}, {q, r}}, {{Vt, 2}, {q, s}, {q, t}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {q}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                        Outpq+=(2.)*Vs[n3*q+n2*r+n*q+s]*V24_2[n3*q+n2*s+n*q+r]*Rho(p,q)*Eta(q,q);
                        //62{{Sign, -1}, {{Vs, 2}, {q, u}, {q, r}}, {{Vt, 2}, {q, s}, {t, q}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {q}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                        Outpq+=(-1.)*Vs[n3*q+n2*r+n*q+s]*V23_2[n3*q+n2*s+n*r+q]*Rho(p,q)*Eta(q,q);
                        //62{{Sign, -1}, {{Vs, 2}, {q, u}, {r, q}}, {{Vt, 2}, {q, s}, {q, t}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {q}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                        Outpq+=(-1.)*Vs[n3*q+n2*r+n*s+q]*V24_2[n3*q+n2*s+n*q+r]*Rho(p,q)*Eta(q,q);
                        //62{{Sign, 1}, {{Vs, 2}, {q, u}, {r, q}}, {{Vt, 2}, {q, s}, {t, q}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {q}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                        Outpq+=(1.)*Vs[n3*q+n2*r+n*s+q]*V23_2[n3*q+n2*s+n*r+q]*Rho(p,q)*Eta(q,q);
                    }
                }
                Out(p,q) += Outpq;
            }
        }
        
        RhoDot_.zeros();
        RhoDot_ += lambda*(Out+Out.t());
        RhoDot_.print("RhoDot");
        
        threading_policy::pop();

    }
    
//*****************************************************************
//  All the junk that follows is dead/nonworking/notmaintained.
//*****************************************************************
    
    // Exponential integrator
    // Work in basis in which L is diagonal, and then move back to the fock eigenbasis.
    // Doesn't work... :(
    void Split_Step_Exp_Euler(const arma::vec& eigval, const arma::cx_mat& Cu , const arma::cx_mat& oldrho, arma::cx_mat& newrho, const double tnow, const double dt, bool& IsOn)
    {
        //        cx_mat Ud = exp(eigval*j*dt);
        //        cx_mat U = Cu*diagmat(Ud)*Cu.t();  // Moves Ud into the fock eigenbasis.
        cx_mat Pp = Cu.t()*oldrho*C;         // Moves P into the basis of F+mu
        cx_mat U(oldrho); U.zeros();
        cx_mat Li(oldrho); Li.zeros();
        cx_mat RhoStep(oldrho); RhoStep.zeros();
        for (int i=0; i<oldrho.n_rows; ++i)
        {
            for (int k=0; k<oldrho.n_rows; ++k)
            {
                if (abs(eigval(i)-eigval(k))>pow(10.0,-11.0))
                {
                    U(i,k) = exp(j*(eigval(i)-eigval(k))*dt);  // in this basis your Shur with this to get U(t)
                    Li(i,k) = 1.0/(j*(eigval(i)-eigval(k)));
                }
                else
                {
                    U(i,k) = 1.0;
                    Li(i,k) = 1.0; // In this basis the diagonal should not evolve under L.
                }
            }
        }
        //        U.print("u");
        //        Li.print("li");
        
        //       RhoDot( oldrho, RhoStep, tnow+(dt/2.0));
        RhoStep = Cu.t()*RhoStep*C; // convert N(rho) into the same basis as the linear parts.
        
        newrho.zeros();
        newrho = Cu*(U%oldrho + Li*(U%RhoStep-RhoStep))*Cu.t(); // Convert back into the orig basis.
        //        newrho.print("NewRho");
        
        // Try it the other way and compare...
        //        Split_RK4_Step_MMUT(eigval,Cu,oldrho,newrho,tnow,dt,IsOn);
        //        newrho.print("newrhoMMut");
        
        
        //y(t) = e^{L t}y_0 + L^{-1} (e^{L t} - 1) \mathcal{N}( y( t_0 ) ).
    }
    
    void Split_Step_Euler(const arma::vec& eigval, const arma::cx_mat& Cu , const arma::cx_mat& oldrho, arma::cx_mat& newrho, const double tnow, const double dt, bool& IsOn)
    {
        cx_mat Ud = exp(eigval*0.5*j*dt);
        cx_mat U = Cu*diagmat(Ud)*Cu.t();
        cx_mat RhoHalfStepped = U*oldrho*U.t();
        newrho.zeros();
        RhoDot(  RhoHalfStepped, newrho,tnow+(dt/2.0));
        newrho = dt*newrho + RhoHalfStepped;
        newrho = U*newrho*U.t();
    }
    
    // Next-Most up-to-date secular term.
    void RhoDotSecFull(const cx_mat& Rho, cx_mat& RhoDot_, const double& time)
    {
        if (params["RIFock"]!=0.0)
            return;
        
        //        Rho.print("Rhodot in");
        
        threading_policy::enable_omp_only();
        cx_mat Eta(Rho); Eta.eye();
        Eta -= Rho;
        cx_mat Out(RhoDot_); Out.zeros();
        cx_mat Out1(RhoDot_); Out1.zeros();
        cx_mat Out2(RhoDot_); Out2.zeros();
        cx_mat Out3(RhoDot_); Out3.zeros();
        cx_mat Out4(RhoDot_); Out4.zeros();
        
        cx_mat d2 = Delta2();
        
        int n=n_mo;
        int n2=n_mo*n_mo;
        int n3=n2*n_mo;
        int n4=n3*n_mo;
        
        typedef std::complex<double> cx;
        std::vector<cx> Vs(n4);
        // These will be contracted against t.
        
        std::fill(Vs.begin(), Vs.end(), std::complex<double>(0.0,0.0));
        std::vector<cx> V13_3(n4), V14_4(n4), V23_3(n4), V24_4(n4), V13_1(n4), V14_1(n4), V23_2(n4), V24_2(n4);
        
        //       {"V13_1", "V13_3", "V14_1", "V14_4", "V23_2", "V23_3", "V24_2", "V24_4"}
        
        std::fill(V13_1.begin(), V13_1.end(), std::complex<double>(0.0,0.0));
        std::fill(V13_3.begin(), V13_3.end(), std::complex<double>(0.0,0.0));
        std::fill(V14_1.begin(), V14_1.end(), std::complex<double>(0.0,0.0));
        std::fill(V14_4.begin(), V14_4.end(), std::complex<double>(0.0,0.0));
        std::fill(V23_2.begin(), V23_2.end(), std::complex<double>(0.0,0.0));
        std::fill(V23_3.begin(), V23_3.end(), std::complex<double>(0.0,0.0));
        std::fill(V24_2.begin(), V24_2.end(), std::complex<double>(0.0,0.0));
        std::fill(V24_4.begin(), V24_4.end(), std::complex<double>(0.0,0.0));
        
#if HASUDR
#pragma omp declare reduction( + : std::vector<cx> : std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< std::complex<double> >( )) ) initializer (omp_priv(omp_orig))
#pragma omp parallel for reduction(+:Vs, V13_3, V14_4, V23_3, V24_4, V13_1, V14_1, V23_2, V24_2) schedule(guided)
#endif
        for(int R=0; R<n_aux; ++R)
        {
            
            cx_mat B = V.t()*BpqR.slice(R)*V;
            cx_mat Bg = B*Rho;
            cx_mat gB = Rho*B;
            cx_mat Be = B*Eta;
            cx_mat eB = Eta*B;
            cx_mat gBe=Rho*Be;
            cx_mat eBg=eB*Rho;
            for(int p=0; p<n_mo; ++p)
            {
                for(int q=0; q<n_mo; ++q)
                {
                    for(int r=0; r<n_mo; ++r)
                    {
                        for(int s=0; s<n_mo; ++s)
                        {
                            std::complex<double> d = (j)/(d2(p,r)+d2(q,s));
                            if(!(d==d and abs(d)<100.0))
                                d=0.0;
                            Vs[p*n3+q*n2+r*n+s] += d*B(p,r)*B(q,s);
                            
                            V13_3[p*n3+q*n2+r*n+s] += gBe(p,r)*B(q,s);
                            V14_4[p*n3+q*n2+r*n+s] += gB(p,r)*Be(q,s);
                            V23_3[p*n3+q*n2+r*n+s] += Be(p,r)*gB(q,s);
                            V24_4[p*n3+q*n2+r*n+s] += B(p,r)*gBe(q,s);
                            
                            V13_1[p*n3+q*n2+r*n+s] += eBg(p,r)*B(q,s);
                            V14_1[p*n3+q*n2+r*n+s] += eB(p,r)*Bg(q,s);
                            V23_2[p*n3+q*n2+r*n+s] += Bg(p,r)*eB(q,s);
                            V24_2[p*n3+q*n2+r*n+s] += B(p,r)*eBg(q,s);
                        }
                    }
                }
            }
        }
#pragma omp barrier
#pragma omp parallel for schedule(guided)
        for(int p=0; p<n_mo; ++p)
        {
            for(int q=0; q<n_mo; ++q)
            {
                for(int r=0; r<n_mo; ++r)
                {
                    for(int s=0; s<n_mo; ++s)
                    {
                        if((Vs[p*n3+q*n2+r*n+s]!=Vs[p*n3+q*n2+r*n+s]) or abs(Vs[p*n3+q*n2+r*n+s])>10.0)
                            Vs[p*n3+q*n2+r*n+s]=0.0;
                    }
                }
            }
        }
#if HASUDR
#pragma omp barrier
#pragma omp declare reduction( + : cx_mat : std::transform(omp_in.begin( ),  omp_in.end( ),  omp_out.begin( ), omp_out.begin( ),  std::plus< std::complex<double> >( )) ) initializer (omp_priv(omp_orig))
#pragma omp parallel for reduction(+:Out,Out1,Out2,Out3,Out4) schedule(guided)
#endif
        for(int p=0; p<n_mo; ++p)
        {
            for(int q=0; q<n_mo; ++q)
            {
                for(int r=0; r<n_mo; ++r)
                {
                    for(int s=0; s<n_mo; ++s)
                    {
                        
                        //61{{Sign, -1}, {{Vs, 2}, {p, u}, {p, r}}, {{Vt, 2}, {p, s}, {p, t}}, {{\[Gamma], 1}, {p}, {p}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                        Out1(p,q)+=(-1.)*Vs[n3*p+n2*r+n*p+s]*V24_2[n3*p+n2*s+n*p+r]*Rho(p,p)*Eta(p,q);
                        //61{{Sign, -1}, {{Vs, 2}, {p, u}, {p, r}}, {{Vt, 2}, {p, s}, {p, t}}, {{\[Gamma], 1}, {p}, {p}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                        Out1(p,q)+=(-1.)*Vs[n3*p+n2*r+n*p+s]*V24_2[n3*p+n2*s+n*p+r]*Rho(p,p)*Eta(p,q);
                        //61{{Sign, 1}, {{Vs, 2}, {p, u}, {p, r}}, {{Vt, 2}, {p, s}, {t, p}}, {{\[Gamma], 1}, {p}, {p}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                        Out1(p,q)+=(1.)*Vs[n3*p+n2*r+n*p+s]*V23_2[n3*p+n2*s+n*r+p]*Rho(p,p)*Eta(p,q);
                        //61{{Sign, 1}, {{Vs, 2}, {p, u}, {r, p}}, {{Vt, 2}, {p, s}, {p, t}}, {{\[Gamma], 1}, {p}, {p}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                        Out1(p,q)+=(1.)*Vs[n3*p+n2*r+n*s+p]*V24_2[n3*p+n2*s+n*p+r]*Rho(p,p)*Eta(p,q);
                        //61{{Sign, -1}, {{Vs, 2}, {p, u}, {r, p}}, {{Vt, 2}, {p, s}, {t, p}}, {{\[Gamma], 1}, {p}, {p}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                        Out1(p,q)+=(-1.)*Vs[n3*p+n2*r+n*s+p]*V23_2[n3*p+n2*s+n*r+p]*Rho(p,p)*Eta(p,q);
                        
                        //62{{Sign, 1}, {{Vs, 2}, {q, u}, {q, r}}, {{Vt, 2}, {q, s}, {q, t}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {q}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                        Out2(p,q)+=(1.)*Vs[n3*q+n2*r+n*q+s]*V24_2[n3*q+n2*s+n*q+r]*Rho(p,q)*Eta(q,q);
                        //62{{Sign, 1}, {{Vs, 2}, {q, u}, {q, r}}, {{Vt, 2}, {q, s}, {q, t}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {q}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                        Out2(p,q)+=(1.)*Vs[n3*q+n2*r+n*q+s]*V24_2[n3*q+n2*s+n*q+r]*Rho(p,q)*Eta(q,q);
                        //62{{Sign, -1}, {{Vs, 2}, {q, u}, {q, r}}, {{Vt, 2}, {q, s}, {t, q}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {q}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                        Out2(p,q)+=(-1.)*Vs[n3*q+n2*r+n*q+s]*V23_2[n3*q+n2*s+n*r+q]*Rho(p,q)*Eta(q,q);
                        //62{{Sign, -1}, {{Vs, 2}, {q, u}, {r, q}}, {{Vt, 2}, {q, s}, {q, t}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {q}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                        Out2(p,q)+=(-1.)*Vs[n3*q+n2*r+n*s+q]*V24_2[n3*q+n2*s+n*q+r]*Rho(p,q)*Eta(q,q);
                        //62{{Sign, 1}, {{Vs, 2}, {q, u}, {r, q}}, {{Vt, 2}, {q, s}, {t, q}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {q}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                        Out2(p,q)+=(1.)*Vs[n3*q+n2*r+n*s+q]*V23_2[n3*q+n2*s+n*r+q]*Rho(p,q)*Eta(q,q);
                        
                        //63{{Sign, -1}, {{Vs, 2}, {q, s}, {q, t}}, {{Vt, 2}, {q, u}, {q, r}}, {{\[Gamma], 1}, {q}, {q}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                        Out3(p,q)+=(-1.)*Vs[n3*q+n2*r+n*q+s]*V24_4[n3*q+n2*s+n*q+r]*Rho(q,q)*Eta(p,q);
                        //63{{Sign, -1}, {{Vs, 2}, {q, s}, {q, t}}, {{Vt, 2}, {q, u}, {q, r}}, {{\[Gamma], 1}, {q}, {q}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                        Out3(p,q)+=(-1.)*Vs[n3*q+n2*r+n*q+s]*V24_4[n3*q+n2*s+n*q+r]*Rho(q,q)*Eta(p,q);
                        //63{{Sign, 1}, {{Vs, 2}, {q, s}, {q, t}}, {{Vt, 2}, {q, u}, {r, q}}, {{\[Gamma], 1}, {q}, {q}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                        Out3(p,q)+=(1.)*Vs[n3*q+n2*r+n*q+s]*V23_3[n3*q+n2*s+n*r+q]*Rho(q,q)*Eta(p,q);
                        //63{{Sign, 1}, {{Vs, 2}, {q, s}, {t, q}}, {{Vt, 2}, {q, u}, {q, r}}, {{\[Gamma], 1}, {q}, {q}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                        Out3(p,q)+=(1.)*Vs[n3*q+n2*r+n*s+q]*V24_4[n3*q+n2*s+n*q+r]*Rho(q,q)*Eta(p,q);
                        //63{{Sign, -1}, {{Vs, 2}, {q, s}, {t, q}}, {{Vt, 2}, {q, u}, {r, q}}, {{\[Gamma], 1}, {q}, {q}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                        Out3(p,q)+=(-1.)*Vs[n3*q+n2*r+n*s+q]*V23_3[n3*q+n2*s+n*r+q]*Rho(q,q)*Eta(p,q);
                        
                        //64{{Sign, 1}, {{Vs, 2}, {p, s}, {p, t}}, {{Vt, 2}, {p, u}, {p, r}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {p}}, {{\[Eta], 1}, {s}, {r}}}
                        Out4(p,q)+=(1.)*Vs[n3*p+n2*r+n*p+s]*V24_4[n3*p+n2*s+n*p+r]*Rho(p,q)*Eta(p,p);
                        //64{{Sign, 1}, {{Vs, 2}, {p, s}, {p, t}}, {{Vt, 2}, {p, u}, {p, r}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {p}}, {{\[Eta], 1}, {s}, {r}}}
                        Out4(p,q)+=(1.)*Vs[n3*p+n2*r+n*p+s]*V24_4[n3*p+n2*s+n*p+r]*Rho(p,q)*Eta(p,p);
                        //64{{Sign, -1}, {{Vs, 2}, {p, s}, {p, t}}, {{Vt, 2}, {p, u}, {r, p}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {p}}, {{\[Eta], 1}, {s}, {r}}}
                        Out4(p,q)+=(-1.)*Vs[n3*p+n2*r+n*p+s]*V23_3[n3*p+n2*s+n*r+p]*Rho(p,q)*Eta(p,p);
                        //64{{Sign, -1}, {{Vs, 2}, {p, s}, {t, p}}, {{Vt, 2}, {p, u}, {p, r}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {p}}, {{\[Eta], 1}, {s}, {r}}}
                        Out4(p,q)+=(-1.)*Vs[n3*p+n2*r+n*s+p]*V24_4[n3*p+n2*s+n*p+r]*Rho(p,q)*Eta(p,p);
                        //64{{Sign, 1}, {{Vs, 2}, {p, s}, {t, p}}, {{Vt, 2}, {p, u}, {r, p}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {p}}, {{\[Eta], 1}, {s}, {r}}}
                        Out4(p,q)+=(1.)*Vs[n3*p+n2*r+n*s+p]*V23_3[n3*p+n2*s+n*r+p]*Rho(p,q)*Eta(p,p);
                        
                        for(int t=0; t<n_mo; ++t)
                        {
                            
                            //7{{Sign, 1}, {{Vs, 2}, {r, v}, {p, s}}, {{Vt, 2}, {r, t}, {p, u}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Gamma], 1}, {r}, {p}}, {{\[Eta], 1}, {t}, {s}}, {{\[Eta], 1}, {r}, {q}}}
                            Out1(p,q)+=(1.)*Vs[n3*r+n2*s+n*p+t]*V24_2[n3*r+n2*t+n*p+s]*Rho(r,p)*Eta(r,q);
                            //71{{Sign, 1}, {{Vs, 2}, {v, r}, {p, s}}, {{Vt, 2}, {t, r}, {p, u}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Gamma], 1}, {r}, {p}}, {{\[Eta], 1}, {t}, {s}}, {{\[Eta], 1}, {r}, {q}}}
                            Out1(p,q)+=(1.)*Vs[n3*r+n2*s+n*p+t]*V14_1[n3*t+n2*s+n*p+r]*Rho(s,p)*Eta(s,q);
                            //71{{Sign, -1}, {{Vs, 2}, {v, r}, {p, s}}, {{Vt, 2}, {t, r}, {u, p}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Gamma], 1}, {r}, {p}}, {{\[Eta], 1}, {t}, {s}}, {{\[Eta], 1}, {r}, {q}}}
                            Out1(p,q)+=(-1.)*Vs[n3*r+n2*s+n*p+t]*V13_1[n3*t+n2*s+n*r+p]*Rho(s,p)*Eta(s,q);
                            //71{{Sign, -1}, {{Vs, 2}, {v, r}, {s, p}}, {{Vt, 2}, {t, r}, {p, u}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Gamma], 1}, {r}, {p}}, {{\[Eta], 1}, {t}, {s}}, {{\[Eta], 1}, {r}, {q}}}
                            Out1(p,q)+=(-1.)*Vs[n3*r+n2*s+n*t+p]*V14_1[n3*t+n2*s+n*p+r]*Rho(s,p)*Eta(s,q);
                            //7{{Sign, 1}, {{Vs, 2}, {v, r}, {s, p}}, {{Vt, 2}, {t, r}, {u, p}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Gamma], 1}, {r}, {p}}, {{\[Eta], 1}, {t}, {s}}, {{\[Eta], 1}, {r}, {q}}}
                            Out1(p,q)+=(1.)*Vs[n3*r+n2*s+n*t+p]*V13_1[n3*t+n2*s+n*r+p]*Rho(s,p)*Eta(s,q);
                            //71{{Sign, 1}, {{Vs, 2}, {v, u}, {p, r}}, {{Vt, 2}, {p, s}, {v, t}}, {{\[Gamma], 1}, {v}, {v}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                            Out1(p,q)+=(1.)*Vs[n3*r+n2*s+n*p+t]*V24_2[n3*p+n2*t+n*r+s]*Rho(r,r)*Eta(p,q);
                            //71{{Sign, 1}, {{Vs, 2}, {u, v}, {p, r}}, {{Vt, 2}, {p, s}, {t, v}}, {{\[Gamma], 1}, {v}, {v}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                            Out1(p,q)+=(1.)*Vs[n3*r+n2*s+n*p+t]*V23_2[n3*p+n2*t+n*r+s]*Rho(s,s)*Eta(p,q);
                            //71{{Sign, 1}, {{Vs, 2}, {v, u}, {p, r}}, {{Vt, 2}, {p, s}, {v, t}}, {{\[Gamma], 1}, {v}, {v}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                            Out1(p,q)+=(1.)*Vs[n3*r+n2*s+n*p+t]*V24_2[n3*p+n2*t+n*r+s]*Rho(r,r)*Eta(p,q);
                            //71{{Sign, -1}, {{Vs, 2}, {v, u}, {p, r}}, {{Vt, 2}, {p, s}, {t, v}}, {{\[Gamma], 1}, {v}, {v}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                            Out1(p,q)+=(-1.)*Vs[n3*r+n2*s+n*p+t]*V23_2[n3*p+n2*t+n*s+r]*Rho(r,r)*Eta(p,q);
                            //71{{Sign, -1}, {{Vs, 2}, {v, u}, {r, p}}, {{Vt, 2}, {p, s}, {v, t}}, {{\[Gamma], 1}, {v}, {v}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                            Out1(p,q)+=(-1.)*Vs[n3*r+n2*s+n*t+p]*V24_2[n3*p+n2*t+n*r+s]*Rho(r,r)*Eta(p,q);
                            //71{{Sign, 1}, {{Vs, 2}, {v, u}, {r, p}}, {{Vt, 2}, {p, s}, {t, v}}, {{\[Gamma], 1}, {v}, {v}}, {{\[Gamma], 1}, {u}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                            Out1(p,q)+=(1.)*Vs[n3*r+n2*s+n*t+p]*V23_2[n3*p+n2*t+n*s+r]*Rho(r,r)*Eta(p,q);
                            
                            
                            //7{{Sign, -1}, {{Vs, 2}, {q, v}, {t, r}}, {{Vt, 2}, {q, s}, {t, u}}, {{\[Gamma], 1}, {p}, {t}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Eta], 1}, {q}, {t}}, {{\[Eta], 1}, {s}, {r}}}
                            Out2(p,q)+=(-1.)*Vs[n3*q+n2*r+n*s+t]*V24_2[n3*q+n2*t+n*s+r]*Rho(p,s)*Eta(q,s);
                            //72{{Sign, -1}, {{Vs, 2}, {q, v}, {r, t}}, {{Vt, 2}, {q, s}, {u, t}}, {{\[Gamma], 1}, {p}, {t}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Eta], 1}, {q}, {t}}, {{\[Eta], 1}, {s}, {r}}}
                            Out2(p,q)+=(-1.)*Vs[n3*q+n2*r+n*s+t]*V23_2[n3*q+n2*s+n*r+t]*Rho(p,t)*Eta(q,t);
                            //72{{Sign, 1}, {{Vs, 2}, {q, v}, {r, t}}, {{Vt, 2}, {q, s}, {t, u}}, {{\[Gamma], 1}, {p}, {t}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Eta], 1}, {q}, {t}}, {{\[Eta], 1}, {s}, {r}}}
                            Out2(p,q)+=(1.)*Vs[n3*q+n2*r+n*s+t]*V24_2[n3*q+n2*s+n*t+r]*Rho(p,t)*Eta(q,t);
                            //72{{Sign, 1}, {{Vs, 2}, {q, v}, {t, r}}, {{Vt, 2}, {q, s}, {u, t}}, {{\[Gamma], 1}, {p}, {t}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Eta], 1}, {q}, {t}}, {{\[Eta], 1}, {s}, {r}}}
                            Out2(p,q)+=(1.)*Vs[n3*q+n2*r+n*s+t]*V23_2[n3*q+n2*t+n*r+s]*Rho(p,s)*Eta(q,s);
                            //72{{Sign, -1}, {{Vs, 2}, {q, v}, {t, r}}, {{Vt, 2}, {q, s}, {t, u}}, {{\[Gamma], 1}, {p}, {t}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Eta], 1}, {q}, {t}}, {{\[Eta], 1}, {s}, {r}}}
                            Out2(p,q)+=(-1.)*Vs[n3*q+n2*r+n*s+t]*V24_2[n3*q+n2*t+n*s+r]*Rho(p,s)*Eta(q,s);
                            //72{{Sign, -1}, {{Vs, 2}, {q, v}, {s, r}}, {{Vt, 2}, {t, r}, {q, u}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Eta], 1}, {t}, {s}}, {{\[Eta], 1}, {r}, {r}}}
                            Out2(p,q)+=(-1.)*Vs[n3*q+n2*r+n*s+t]*V14_1[n3*s+n2*t+n*q+r]*Rho(p,q)*Eta(t,t);
                            //72{{Sign, -1}, {{Vs, 2}, {q, v}, {r, s}}, {{Vt, 2}, {r, t}, {q, u}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Eta], 1}, {t}, {s}}, {{\[Eta], 1}, {r}, {r}}}
                            Out2(p,q)+=(-1.)*Vs[n3*q+n2*r+n*s+t]*V24_2[n3*s+n2*t+n*q+r]*Rho(p,q)*Eta(s,s);
                            //72{{Sign, -1}, {{Vs, 2}, {q, v}, {s, r}}, {{Vt, 2}, {t, r}, {q, u}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Eta], 1}, {t}, {s}}, {{\[Eta], 1}, {r}, {r}}}
                            Out2(p,q)+=(-1.)*Vs[n3*q+n2*r+n*s+t]*V14_1[n3*s+n2*t+n*q+r]*Rho(p,q)*Eta(t,t);
                            //72{{Sign, 1}, {{Vs, 2}, {q, v}, {s, r}}, {{Vt, 2}, {t, r}, {u, q}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Eta], 1}, {t}, {s}}, {{\[Eta], 1}, {r}, {r}}}
                            Out2(p,q)+=(1.)*Vs[n3*q+n2*r+n*s+t]*V13_1[n3*s+n2*t+n*r+q]*Rho(p,q)*Eta(t,t);
                            //72{{Sign, 1}, {{Vs, 2}, {q, v}, {r, s}}, {{Vt, 2}, {t, r}, {q, u}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Eta], 1}, {t}, {s}}, {{\[Eta], 1}, {r}, {r}}}
                            Out2(p,q)+=(1.)*Vs[n3*q+n2*r+n*s+t]*V14_1[n3*t+n2*s+n*q+r]*Rho(p,q)*Eta(s,s);
                            //72{{Sign, -1}, {{Vs, 2}, {q, v}, {r, s}}, {{Vt, 2}, {t, r}, {u, q}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Eta], 1}, {t}, {s}}, {{\[Eta], 1}, {r}, {r}}}
                            Out2(p,q)+=(-1.)*Vs[n3*q+n2*r+n*s+t]*V13_1[n3*t+n2*s+n*r+q]*Rho(p,q)*Eta(s,s);
                            
                            
                            //7{{Sign, 1}, {{Vs, 2}, {q, s}, {t, u}}, {{Vt, 2}, {q, v}, {t, r}}, {{\[Gamma], 1}, {q}, {t}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Eta], 1}, {p}, {t}}, {{\[Eta], 1}, {s}, {r}}}
                            Out3(p,q)+=(1.)*Vs[n3*q+n2*r+n*s+t]*V24_4[n3*q+n2*t+n*s+r]*Rho(q,s)*Eta(p,s);
                            //73{{Sign, 1}, {{Vs, 2}, {q, s}, {u, t}}, {{Vt, 2}, {q, v}, {r, t}}, {{\[Gamma], 1}, {q}, {t}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Eta], 1}, {p}, {t}}, {{\[Eta], 1}, {s}, {r}}}
                            Out3(p,q)+=(1.)*Vs[n3*q+n2*r+n*s+t]*V23_3[n3*q+n2*s+n*r+t]*Rho(q,t)*Eta(p,t);
                            //73{{Sign, -1}, {{Vs, 2}, {q, s}, {u, t}}, {{Vt, 2}, {q, v}, {t, r}}, {{\[Gamma], 1}, {q}, {t}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Eta], 1}, {p}, {t}}, {{\[Eta], 1}, {s}, {r}}}
                            Out3(p,q)+=(-1.)*Vs[n3*q+n2*r+n*s+t]*V24_4[n3*q+n2*s+n*t+r]*Rho(q,t)*Eta(p,t);
                            //73{{Sign, -1}, {{Vs, 2}, {q, s}, {t, u}}, {{Vt, 2}, {q, v}, {r, t}}, {{\[Gamma], 1}, {q}, {t}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Eta], 1}, {p}, {t}}, {{\[Eta], 1}, {s}, {r}}}
                            Out3(p,q)+=(-1.)*Vs[n3*q+n2*r+n*s+t]*V23_3[n3*q+n2*t+n*r+s]*Rho(q,s)*Eta(p,s);
                            //73{{Sign, 1}, {{Vs, 2}, {q, s}, {t, u}}, {{Vt, 2}, {q, v}, {t, r}}, {{\[Gamma], 1}, {q}, {t}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Eta], 1}, {p}, {t}}, {{\[Eta], 1}, {s}, {r}}}
                            Out3(p,q)+=(1.)*Vs[n3*q+n2*r+n*s+t]*V24_4[n3*q+n2*t+n*s+r]*Rho(q,s)*Eta(p,s);
                            //73{{Sign, 1}, {{Vs, 2}, {q, s}, {u, t}}, {{Vt, 2}, {v, t}, {q, r}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                            Out3(p,q)+=(1.)*Vs[n3*q+n2*r+n*s+t]*V14_4[n3*s+n2*t+n*q+r]*Rho(t,t)*Eta(p,q);
                            //73{{Sign, 1}, {{Vs, 2}, {q, s}, {t, u}}, {{Vt, 2}, {t, v}, {q, r}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                            Out3(p,q)+=(1.)*Vs[n3*q+n2*r+n*s+t]*V24_4[n3*s+n2*t+n*q+r]*Rho(s,s)*Eta(p,q);
                            //73{{Sign, 1}, {{Vs, 2}, {q, s}, {u, t}}, {{Vt, 2}, {v, t}, {q, r}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                            Out3(p,q)+=(1.)*Vs[n3*q+n2*r+n*s+t]*V14_4[n3*s+n2*t+n*q+r]*Rho(t,t)*Eta(p,q);
                            //73{{Sign, -1}, {{Vs, 2}, {q, s}, {u, t}}, {{Vt, 2}, {v, t}, {r, q}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                            Out3(p,q)+=(-1.)*Vs[n3*q+n2*r+n*s+t]*V13_3[n3*s+n2*t+n*r+q]*Rho(t,t)*Eta(p,q);
                            //73{{Sign, -1}, {{Vs, 2}, {q, s}, {t, u}}, {{Vt, 2}, {v, t}, {q, r}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                            Out3(p,q)+=(-1.)*Vs[n3*q+n2*r+n*s+t]*V14_4[n3*t+n2*s+n*q+r]*Rho(s,s)*Eta(p,q);
                            //73{{Sign, 1}, {{Vs, 2}, {q, s}, {t, u}}, {{Vt, 2}, {v, t}, {r, q}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Gamma], 1}, {t}, {t}}, {{\[Eta], 1}, {p}, {q}}, {{\[Eta], 1}, {s}, {r}}}
                            Out3(p,q)+=(1.)*Vs[n3*q+n2*r+n*s+t]*V13_3[n3*t+n2*s+n*r+q]*Rho(s,s)*Eta(p,q);
                            
                            
                            //7{{Sign, -1}, {{Vs, 2}, {r, t}, {p, u}}, {{Vt, 2}, {r, v}, {p, s}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Gamma], 1}, {r}, {q}}, {{\[Eta], 1}, {t}, {s}}, {{\[Eta], 1}, {r}, {p}}}
                            Out4(p,q)+=(-1.)*Vs[n3*r+n2*s+n*p+t]*V24_4[n3*r+n2*t+n*p+s]*Rho(r,q)*Eta(r,p);
                            //74{{Sign, -1}, {{Vs, 2}, {t, r}, {p, u}}, {{Vt, 2}, {v, r}, {p, s}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Gamma], 1}, {r}, {q}}, {{\[Eta], 1}, {t}, {s}}, {{\[Eta], 1}, {r}, {p}}}
                            Out4(p,q)+=(-1.)*Vs[n3*r+n2*s+n*p+t]*V14_4[n3*t+n2*s+n*p+r]*Rho(s,q)*Eta(s,p);
                            //74{{Sign, 1}, {{Vs, 2}, {t, r}, {p, u}}, {{Vt, 2}, {v, r}, {s, p}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Gamma], 1}, {r}, {q}}, {{\[Eta], 1}, {t}, {s}}, {{\[Eta], 1}, {r}, {p}}}
                            Out4(p,q)+=(1.)*Vs[n3*r+n2*s+n*p+t]*V13_3[n3*t+n2*s+n*r+p]*Rho(s,q)*Eta(s,p);
                            //74{{Sign, 1}, {{Vs, 2}, {t, r}, {u, p}}, {{Vt, 2}, {v, r}, {p, s}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Gamma], 1}, {r}, {q}}, {{\[Eta], 1}, {t}, {s}}, {{\[Eta], 1}, {r}, {p}}}
                            Out4(p,q)+=(1.)*Vs[n3*r+n2*s+n*t+p]*V14_4[n3*t+n2*s+n*p+r]*Rho(s,q)*Eta(s,p);
                            //74{{Sign, -1}, {{Vs, 2}, {t, r}, {u, p}}, {{Vt, 2}, {v, r}, {s, p}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Gamma], 1}, {r}, {q}}, {{\[Eta], 1}, {t}, {s}}, {{\[Eta], 1}, {r}, {p}}}
                            Out4(p,q)+=(-1.)*Vs[n3*r+n2*s+n*t+p]*V13_3[n3*t+n2*s+n*r+p]*Rho(s,q)*Eta(s,p);
                            //74{{Sign, -1}, {{Vs, 2}, {t, s}, {p, u}}, {{Vt, 2}, {p, v}, {t, r}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Eta], 1}, {t}, {t}}, {{\[Eta], 1}, {s}, {r}}}
                            Out4(p,q)+=(-1.)*Vs[n3*r+n2*s+n*p+t]*V24_4[n3*p+n2*t+n*r+s]*Rho(p,q)*Eta(r,r);
                            //74{{Sign, -1}, {{Vs, 2}, {s, t}, {p, u}}, {{Vt, 2}, {p, v}, {r, t}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Eta], 1}, {t}, {t}}, {{\[Eta], 1}, {s}, {r}}}
                            Out4(p,q)+=(-1.)*Vs[n3*r+n2*s+n*p+t]*V23_3[n3*p+n2*t+n*r+s]*Rho(p,q)*Eta(s,s);
                            //74{{Sign, -1}, {{Vs, 2}, {t, s}, {p, u}}, {{Vt, 2}, {p, v}, {t, r}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Eta], 1}, {t}, {t}}, {{\[Eta], 1}, {s}, {r}}}
                            Out4(p,q)+=(-1.)*Vs[n3*r+n2*s+n*p+t]*V24_4[n3*p+n2*t+n*r+s]*Rho(p,q)*Eta(r,r);
                            //74{{Sign, 1}, {{Vs, 2}, {t, s}, {p, u}}, {{Vt, 2}, {p, v}, {r, t}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Eta], 1}, {t}, {t}}, {{\[Eta], 1}, {s}, {r}}}
                            Out4(p,q)+=(1.)*Vs[n3*r+n2*s+n*p+t]*V23_3[n3*p+n2*t+n*s+r]*Rho(p,q)*Eta(r,r);
                            //74{{Sign, 1}, {{Vs, 2}, {t, s}, {u, p}}, {{Vt, 2}, {p, v}, {t, r}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Eta], 1}, {t}, {t}}, {{\[Eta], 1}, {s}, {r}}}
                            Out4(p,q)+=(1.)*Vs[n3*r+n2*s+n*t+p]*V24_4[n3*p+n2*t+n*r+s]*Rho(p,q)*Eta(r,r);
                            //74{{Sign, -1}, {{Vs, 2}, {t, s}, {u, p}}, {{Vt, 2}, {p, v}, {r, t}}, {{\[Gamma], 1}, {p}, {q}}, {{\[Gamma], 1}, {v}, {u}}, {{\[Eta], 1}, {t}, {t}}, {{\[Eta], 1}, {s}, {r}}}
                            Out4(p,q)+=(-1.)*Vs[n3*r+n2*s+n*t+p]*V23_3[n3*p+n2*t+n*s+r]*Rho(p,q)*Eta(r,r);
                        }
                    }
                }
            }
        }
#pragma omp barrier
        RhoDot_.zeros();
        
        // Check positivity.
        bool Positive=true;
        for (int i=0; i<n_mo; ++i)
        {
            if(real(Rho(i,i)+0.02*RhoDot_(i,i))<0)
            {
                cout << " Nonpositive " << i  << Rho(i,i) << RhoDot_(i,i) << Rho(i,i)+0.02*RhoDot_(i,i) << endl;
                Positive = false;
            }
        }
        if (false && !Positive)
        {
            Out1.print("Out1");
            Out2.print("Out2");
            Out3.print("Out3");
            Out4.print("Out4");
            cout << "d(trace)" << trace(Out1) << endl;
            cout << "d(trace)" << trace(Out2) << endl;
            cout << "d(trace)" << trace(Out3) << endl;
            cout << "d(trace)" << trace(Out4) << endl;
        }
        
        Out = (Out1+Out2+Out3+Out4);
        //        Out.print("Out");
        if (abs(accu((Out - Out.t())))>pow(10.0,-10.0))
            cout << "HERMITICITY ERROR" << abs(accu((Out - Out.t()))) << endl;
        
        RhoDot_ += 0.5*(Out+Out.t()); // Small numerical errors can accumulate.
        RhoDot_.print("RhoDot(slow)");
        cout << "d(trace)" << trace(RhoDot_) << endl;
        
        
        //      RhoDot_ -= Out.t();
        threading_policy::pop();
        /*
         Rho.print("Rho");
         RhoDot_.print("RhoDot");
         */
    }
    
    
};

#endif