#ifndef TCLproph
#define TCLproph

#include <iostream>
#include <fstream>
#include <sstream>
#include <armadillo>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include "TCL.h"
#include "TCL_EE2.h"
#include "math.h"
#include "rks.h"

using namespace arma;

#define eVPerAu 27.2113
#define FsPerAu 0.0241888

inline bool FileExists(const char* name)
{
	bool found= false;
	try{
		FILE* file;
		found = (file = fopen(name, "r"));
	}
	catch(...)
	{ }
	return found;
}

class TDSCF
{
public:
	int n_ao, n_mo;
	int n_occ;
    int n_aux;
	int n, n_e;
	int n_fock;
    int loop, loop1, loop2, loop3;
    int firstOrb, lastOrb;
    int npol_col;
    double ECore,Entropy,Energy; // Energy of core density.
    double t;
    long double wallt0;
    
    rks* the_scf;
    TCLMatrices* MyTCL;
	EE2* ee2;
    FieldMatrices* Mus;
    TDSCF_Temporaries* Scratch;
    std::map<std::string,double> params;
    
    // For TDSSCF
    vec f_diag;
    cx_mat C;
    cx_mat HJK; // For incfock.
    mat P0;
    vec f_diagb;
    cx_mat V0,V; // (lowdinXFock change of basis.)
    cx_mat Vs,Csub; // (lowdinXFock change of basis for a sub-matrix.)
    // All these quantities are in the lowdin (x-basis) --------
    cx_mat P0cx;  // Initial core density matrix (x-basis)
    cx_mat P_proj; // Projector onto 'core' initial density
    cx_mat Q_proj; // Complement of 'core' initial density.
    cx_mat Ps_proj; // Projector onto 'core' initial density (rectangular)
    cx_mat Qs_proj; // Complement of 'core' initial density. (rectangular)
    cx_mat sH,sHJK; // Small H and HJK matrices in case update fock is off.
	cx_mat old_fock; // For incremental fock build
	cx_mat old_p;
    cx_mat Rho,NewRho,RhoM12;
    mat Pol;
    cx_vec LastFockPopulations;
    vec LastFockEnergies;
    arma::mat Pops; arma::mat FockEs;
    
	// Parameters ---------
    std::string logprefix;
    // Parameters of any propagation. --------
	bool RunDynamics;
	double dt;
	int MaxIter;
	double RateBuildThresh;
	bool ActiveSpace;
	int ActiveFirst;
    int ActiveLast;
	double ActiveCutoff;
	bool UpdateFockMatrix;
	bool IncFock;
	int UpdateEvery;
	bool UpdateMarkovRates;
    int Stabilize;
	int InitialCondition; // Depreciate?
    bool Restart;
	bool ApplyImpulse;
    bool ApplyNoOscField; // You can apply a static field.
	bool ApplyCw;
	double FieldFreq;
	double FieldAmplitude;
	vec FieldVector;
    double Tau; // PulseLength
	double tOn; // cental time of gaussian pulse.
    // Relaxation Conditions ---------
    double Temp;
	bool TCLOn;
    //--- Correlated Dynamics theory.
    bool Corr;
    bool RIFock;
    bool Mp2NOGuess;
    //--- Pump Probe.
    int Stutter;
    int StartFromSetman;
	bool ApplyImpulse2; // These are unused normally owing to TDDFT's nonstationarity.
	double FieldFreq2;
	double FieldAmplitude2;
	double Tau2;
	double tOn2;
	// Logging ---------
    int SerialNumber;
	int DipolesEvery;
	int StatusEvery;
	int SaveFockEvery;
	int Print;
	bool SaveDipoles;
	bool SavePopulations;
	bool SaveFockEnergies;
	int WriteDensities;
	int SaveEvery;
	int FourierEvery;
	double FourierZoom;
	
    // Argument over-rides any defaults or params read from disk.
    std::map<std::string,double> ReadParameters(std::map<std::string,double> ORParams_ = std::map<std::string,double>())
	{
		// Set defaults.
        SerialNumber = 0;
        Stutter = 0;
        StartFromSetman = 0;
        RunDynamics = true;
        Corr = false;
        RIFock = false;
        Mp2NOGuess = true;
        dt = 0.02;
		MaxIter = 10000;
		RateBuildThresh = 0.001;
		ActiveSpace = false;
		ActiveCutoff = 15.0/27.2113; // the tcl propagation is sometimes unstable with more than 30 electrons.
        ActiveFirst = 0;
        ActiveLast = 0;
		UpdateFockMatrix = true; // Update the fock matrix during relaxation.
		UpdateEvery = 1;
		IncFock = false;
		UpdateMarkovRates = true; // Update the markov rates when the eigenvalues change enough.
		Stabilize = 2;
        TCLOn = false;
		// Initial Conditions.
        Temp = 300.0;
		InitialCondition = 0; // 0=Mean Field Solution. 1=Particle hole Excitation
		Restart = false;
		ApplyImpulse = true;
		ApplyCw = false;
        ApplyNoOscField = false;
		FieldFreq = 1.1;
        Tau = 0.07;
		tOn = 7.0*Tau;
		FieldAmplitude = 0.001;
		FieldVector = vec(3); FieldVector.zeros(); FieldVector(1)=1.0;
		// Second pulse.
		ApplyImpulse2= false;
		FieldFreq2 = 3.0/27.2113;
		FieldAmplitude2 = 0.001;
		Tau2 = 0.07;
		tOn2 = tOn+10.0/0.024;
		// Logging
        DipolesEvery = 2;
		StatusEvery = 15;
		Print = 0;
		SaveDipoles = true;
		SavePopulations = false; // Save energy eigenbasis populations.
		SaveFockEnergies = false; // Save energy eigenvalues during a relaxation process.
		SaveFockEvery = 100;
		SaveEvery = 200; // In units of dt
		WriteDensities = 0; // 1=Write phaseless density (multiple times in one file for Jmol)  2=Write Attch/Detach (for gabedit)
		FourierEvery = 5000;
        FourierZoom = 0.269*(DipolesEvery*dt);

        std::map<std::string,double> vals;
		if (!FileExists("TDSCF.prm"))
		{
			cout << "===============================================" << endl;
			cout << "Didn't find parameters file. Using defaults!!! " << endl;
			cout << "===============================================" << endl;
		}
		else
		{
			ifstream prms("TDSCF.prm");
			std::string key; double val;
			prms.seekg(0);
			if(prms.is_open())
			{
				while(!prms.eof())
				{
					prms >> key >> val;
					vals[key]=val;
				}
			}
			else
			{
				cout << "Couldn't open Parameters :( ... " << endl;
				throw 1;
			}
			prms.close();
            
            ORParams_.insert(vals.begin(), vals.end()); // Passed parameters over-ride defaults and TDSCF.prm.
            std::swap(vals,ORParams_);
            
            // The fact that this is the fewest-lines way to do this in C++ is a simple example of what a failure the language was before x11 which sadly qchem will not compile under.
            Stutter = ((vals.find("Stutter") != vals.end())? vals.find("Stutter")->second : Stutter);
			SerialNumber = ((vals.find("SerialNumber") != vals.end())? vals.find("SerialNumber")->second : SerialNumber);
			StartFromSetman = ((vals.find("StartFromSetman") != vals.end())? vals.find("StartFromSetman")->second : StartFromSetman);
			RunDynamics = ((vals.find("RunDynamics") != vals.end())? vals.find("RunDynamics")->second : RunDynamics);
			Corr = ((vals.find("Corr") != vals.end())? vals.find("Corr")->second : Corr);
            RIFock = ((vals.find("RIFock") != vals.end())? vals.find("RIFock")->second : RIFock);
            if ((vals.find("GML") == vals.end()))
                vals["GML"] = 0.0;
            else
                vals["GML"] = vals.find("GML")->second;
            if ((vals.find("FD") == vals.end()))
                vals["FD"] = 0.0;
            else
                vals["FD"] = vals.find("FD")->second;
            if ((vals.find("DumpEE2") == vals.end()))
                vals["DumpEE2"] = 0.0;
            else
                vals["DumpEE2"] = vals.find("FD")->second;
            Mp2NOGuess = ((vals.find("Mp2NOGuess") != vals.end())? vals.find("Mp2NOGuess")->second : Mp2NOGuess);
            dt = ((vals.find("dt") != vals.end())? vals.find("dt")->second : dt);
			RateBuildThresh = ((vals.find("RateBuildThresh") != vals.end())? vals.find("RateBuildThresh")->second : RateBuildThresh);
			ActiveSpace = ((vals.find("ActiveSpace") != vals.end())? vals.find("ActiveSpace")->second : ActiveSpace);
			ActiveFirst = ((vals.find("ActiveFirst") != vals.end())? vals.find("ActiveFirst")->second : ActiveFirst);
			ActiveLast = ((vals.find("ActiveLast") != vals.end())? vals.find("ActiveLast")->second : ActiveLast);
			Restart = ((vals.find("Restart") != vals.end())? vals.find("Restart")->second : Restart);
            if ((vals.find("CanRestart") == vals.end()))
                vals["CanRestart"] = 0.0;
            Stabilize = ((vals.find("Stabilize") != vals.end())? vals.find("Stabilize")->second : Stabilize);
			ActiveCutoff = ((vals.find("ActiveCutoff") != vals.end())? vals.find("ActiveCutoff")->second : ActiveCutoff);
			UpdateFockMatrix = ((vals.find("UpdateFockMatrix") != vals.end())? vals.find("UpdateFockMatrix")->second : UpdateFockMatrix);
			UpdateEvery = ((vals.find("UpdateEvery") != vals.end())? vals.find("UpdateEvery")->second : UpdateEvery);
			IncFock = ((vals.find("IncFock") != vals.end())? vals.find("IncFock")->second : IncFock);
			UpdateMarkovRates = ((vals.find("UpdateMarkovRates") != vals.end())? vals.find("UpdateMarkovRates")->second : UpdateMarkovRates);
			TCLOn = ((vals.find("TCLOn") != vals.end())? vals.find("TCLOn")->second : TCLOn);
            Temp = ((vals.find("Temp") != vals.end())? vals.find("Temp")->second : Temp);
			MaxIter = ((vals.find("MaxIter") != vals.end())? vals.find("MaxIter")->second : MaxIter);
			InitialCondition = ((vals.find("InitialCondition") != vals.end())? vals.find("InitialCondition")->second : InitialCondition);
			ApplyImpulse = ((vals.find("ApplyImpulse") != vals.end())? vals.find("ApplyImpulse")->second : ApplyImpulse);
			ApplyCw = ((vals.find("ApplyCw") != vals.end())? vals.find("ApplyCw")->second : ApplyCw);
			FieldFreq = ((vals.find("FieldFreq") != vals.end())? vals.find("FieldFreq")->second : FieldFreq);
			Tau = ((vals.find("Tau") != vals.end())? vals.find("Tau")->second : Tau);
			tOn = ((vals.find("tOn") != vals.end())? vals.find("tOn")->second : tOn);
			FieldAmplitude = ((vals.find("FieldAmplitude") != vals.end())? vals.find("FieldAmplitude")->second : FieldAmplitude);
			FieldVector(0) = ((vals.find("ExDir") != vals.end())? vals.find("ExDir")->second : FieldVector(0));
            FieldVector(1) = ((vals.find("EyDir") != vals.end())? vals.find("EyDir")->second : FieldVector(1));
            FieldVector(2) = ((vals.find("EzDir") != vals.end())? vals.find("EzDir")->second : FieldVector(2));
			ApplyImpulse2 = ((vals.find("ApplyImpulse2") != vals.end())? vals.find("ApplyImpulse2")->second : ApplyImpulse2);
			ApplyNoOscField = ((vals.find("ApplyNoOscField") != vals.end())? vals.find("ApplyNoOscField")->second : ApplyNoOscField);
            FieldFreq2 = ((vals.find("FieldFreq2") != vals.end())? vals.find("FieldFreq2")->second : FieldFreq2);
			FieldAmplitude2 = ((vals.find("FieldAmplitude2") != vals.end())? vals.find("FieldAmplitude2")->second : FieldAmplitude2);
			Tau2 = ((vals.find("Tau2") != vals.end())? vals.find("Tau2")->second : Tau2);
			tOn2 = ((vals.find("tOn2") != vals.end())? vals.find("tOn2")->second : FieldFreq);
			StatusEvery = ((vals.find("StatusEvery") != vals.end())? vals.find("StatusEvery")->second : StatusEvery);
			Print = ((vals.find("Print") != vals.end())? vals.find("Print")->second : Print);
			SaveDipoles = ((vals.find("SaveDipoles") != vals.end())? vals.find("SaveDipoles")->second : SaveDipoles);
			SavePopulations = ((vals.find("SavePopulations") != vals.end())? vals.find("SavePopulations")->second : SavePopulations);
            if ((vals.find("PrntPopulationType") == vals.end()))
                vals["PrntPopulationType"] = 0.0; // 0 = Populations of fock orbitals. 1= Natural orbital populations.
            if ((vals.find("SavePopulationsEvery") == vals.end()))
                vals["SavePopulationsEvery"] = 100.0;
            if ((vals.find("TSample") == vals.end()))
                vals["TSample"] = 1000.0;
            if ((vals.find("TWarm") == vals.end()))
                vals["TWarm"] = 1.0;
            if ((vals.find("ExcitedFraction") == vals.end()))
                vals["ExcitedFraction"] = 1.0;
			SaveFockEnergies = ((vals.find("SaveFockEnergies") != vals.end())? vals.find("SaveFockEnergies")->second : SaveFockEnergies);
			SaveFockEvery = ((vals.find("SaveFockEvery") != vals.end())? vals.find("SaveFockEvery")->second : SaveFockEvery);
			WriteDensities = ((vals.find("WriteDensities") != vals.end())? vals.find("WriteDensities")->second : WriteDensities);
			SaveEvery = ((vals.find("SaveEvery") != vals.end())? vals.find("SaveEvery")->second : SaveEvery);
			DipolesEvery = ((vals.find("DipolesEvery") != vals.end())? vals.find("DipolesEvery")->second : DipolesEvery);
			FourierEvery = ((vals.find("FourierEvery") != vals.end())? vals.find("FourierEvery")->second : FourierEvery);
            FourierZoom = ((vals.find("FourierZoom") != vals.end())? vals.find("FourierZoom")->second :
                           0.134524*(DipolesEvery*dt));
		}
        
        // Set up separate logging prefixes if there is more than one propagation going on.
        if (SerialNumber!=0)
        {
            char str[12];
            sprintf(str, "%i", SerialNumber);
            std::string suffix(str);
            logprefix=logprefix+std::string("S")+suffix+std::string("_");
        }
        
		cout << "CONDITIONS:-----------" << endl;
        cout << "RIFock: " << RIFock << endl;
        if (Corr)
        {
            cout << "Corr: " << Corr << endl;
            cout << "GML (Gell-Man Low): " << vals["GML"] << endl;
            cout << "Mp2NOGuess: " << Mp2NOGuess << endl;
        }
        cout << "TCLOn: " << TCLOn << endl;
        cout << "Temp: " << Temp << endl;
        if (TCLOn)
        {
            cout << "UpdateMarkovRates: " << UpdateMarkovRates << endl;
            cout << "RateBuildThresh: " << RateBuildThresh << endl;
        }
		cout << "InitialCondition: " << InitialCondition << endl;
		cout << "StartFromSetman: " << StartFromSetman << endl;
		cout << "MaxIter: " << MaxIter << endl;
		cout << "Restart: " << Restart << endl;
		cout << "FieldAmplitude: " << FieldAmplitude << endl;
		cout << "ApplyImpulse: " << ApplyImpulse << endl;
		cout << "ApplyCw: " << ApplyCw << endl;
		cout << "FieldFreq: " << FieldFreq << endl;
		cout << "Tau: " << Tau << endl;
		cout << "tOn: " << tOn << endl;
		FieldVector.st().print("FieldVector");
        if (ApplyNoOscField)
        {
            cout << "ApplyNoOscField: " << ApplyNoOscField << endl;
            cout << "Tau: " << Tau << endl;
            cout << "tOn: " << tOn << endl;
        }
		if (ApplyImpulse2)
		{
			cout << "ApplyImpulse2: " << ApplyImpulse2 << endl;
			cout << "FieldAmplitude2: " << FieldAmplitude2 << endl;
			cout << "FieldFreq2: " << FieldFreq2 << endl;
			cout << "Tau2: " << Tau2 << endl;
			cout << "tOn2: " << tOn2 << endl;
		}
        
        cout << "THRESHOLDS: -----------" << endl;
		cout << "dt: " << dt << endl;
		cout << "ActiveSpace: " << ActiveSpace << endl;
        if (ActiveLast!=0)
        {
            cout << "ActiveFirst: " << ActiveFirst << endl;
            cout << "ActiveLast: " << ActiveLast << endl;
        }
        else
            cout << "ActiveSpaceCutoff: " << ActiveCutoff << endl;
		cout << "UpdateFockMatrix: " << UpdateFockMatrix << endl;
        if (UpdateFockMatrix && UpdateEvery!=1)
            cout << "UpdateFockMatrix at (n) Steps: " << UpdateEvery << endl;
		cout << "IncFock : " << IncFock << endl;
		cout << "Stabilize: " << Stabilize << endl;
        
		cout << "LOGGING:--------------" << endl;
        cout << "SerialNumber: " << SerialNumber << endl;
        cout << "logprefix: " << logprefix << endl;
		cout << "Stutter: " << Stutter << endl;
        cout << "StatusEvery: " << StatusEvery << endl;
		cout << "DipolesEvery: " << DipolesEvery << endl;
		cout << "Print: " << Print << endl;
		cout << "SaveDipoles: " << SaveDipoles << endl;
		cout << "SaveEvery: " << SaveEvery << endl;
		cout << "WriteDensities: " << WriteDensities << endl;
		cout << "FourierEvery: " << FourierEvery << endl;
		cout << "FourierZoom: " << FourierZoom << endl;
		cout << "SavePopulations: " << SavePopulations << endl;
        cout << "SavePopulationsEvery" << vals["SavePopulationsEvery"] << endl;
		cout << "SaveFockEnergies: " << SaveFockEnergies << endl;
		cout << "SaveFockEvery: " << SaveFockEvery << endl;
		cout << "======================" << endl << endl;
        return vals;
	}
	
    void SetupActiveSpace()
    {
        if (ActiveFirst==0)
        {
            for (; firstOrb<n; ++firstOrb)
                if (abs(f_diagb(firstOrb)-f_diagb(n_occ-1))<ActiveCutoff)
                    break;
            lastOrb = firstOrb+1;
            for (; lastOrb<n; ++lastOrb)
                if (abs(f_diagb(lastOrb)-f_diagb(n_occ-1))>ActiveCutoff)
                    break;
            if (lastOrb < n_occ + 3)
                lastOrb = min(n,lastOrb+8);
            cout << "Propagating within the sub-matrix: " << firstOrb << " to " << lastOrb << " (inclusive)"  << endl;
            if (lastOrb == n)
                lastOrb = n-1;
            cout << "Propagating within the sub-matrix: " << f_diagb(firstOrb) << " (eH) to " << f_diagb(lastOrb) << " eH (inclusive)"  << endl;
            cout << "Propagating within the sub-matrix: " << 27.2113*f_diagb(firstOrb) << " (eV) to " << 27.2113*f_diagb(lastOrb) << " eV (inclusive)"  << endl;
        }
        else
        {
            firstOrb = ActiveFirst;
            lastOrb = ActiveLast;
        }
        
        int nact = lastOrb-firstOrb+1;
        
        // Generate projectors these are in the Lowdin (X) basis.
        cx_mat P_(n,n); P_.eye();
        P0cx=P_;
        P0cx.submat(firstOrb,firstOrb,n-1,n-1) *= 0;
        //P0cx is the core density matrix in the lowdin basis.
        P0cx = V*P0cx*V.t(); // Lowdin basis core density matrix.
        P_.submat(firstOrb,firstOrb,lastOrb,lastOrb) *= 0.0;
        P_proj = V*P_*V.t(); // Lowdin basis core inactive projector
        Q_proj = eye<cx_mat>(n,n) - V*P_*V.t(); // Lowdin basis active projector
        // Make the rectangular versions too:
        Ps_proj.zeros(); Qs_proj.zeros();
        Ps_proj.resize(n,n-nact); // projector onto inactive space (low X fock)
        int j=0;
        for (int i=0;i<n;++i)
        {
            if (i<firstOrb or i>lastOrb)
            {
                Ps_proj.col(j) = V.col(i);
                ++j;
            }
        }
        Qs_proj.resize(n,nact); // projector onto active space (low X fock)
        j=0;
        for (int i=0;i<n;++i)
        {
            if (i>=firstOrb and i<=lastOrb)
            {
                Qs_proj.col(j) = V.col(i);
                ++j;
            }
        }
        cout << "N Core St: " << trace(real(P_proj)) << " N Active St:" << trace(real(Q_proj)) << endl;
        P_.eye(); P_.submat(the_scf->NOcc,the_scf->NOcc,n-1,n-1) *= 0.0; cx_mat Px = V*P_*V.t();
        cout << "Tr(Ps Core-Space)" << trace(Ps_proj.t()*Px*Ps_proj) << " Tr(Ps Active Space)" << trace(Qs_proj.t()*Px*Qs_proj) << endl;
        cout << " Tr(Pc Active Space) (should be zero) " << trace(Qs_proj.t()*P0cx*Qs_proj) << endl;
        f_diag = f_diagb.subvec(firstOrb,lastOrb);
        n = lastOrb-firstOrb+1;
        n_e = n_occ - firstOrb;
        Vs = Qs_proj; // LAO->current eigenbasis.
        Csub = the_scf->X*Vs;
        old_fock.zeros();
        old_p.zeros();
    }
    
    void PlotFields()
    {
        cx_mat Field(n_mo,n_mo); Field.fill(0.0);
        double x=10.0; bool y = false;
        mat FieldR = real(Mus->mux);
        //FieldR.print("Real Field");
        vec eigval; mat Cu;
        eig_sym(eigval,Cu,FieldR);//*the_scf->AOS);
        //eigval.print("Eigval");
        String fileName  = String("./Perturbation.molden");
        FILE *fp = fopen(fileName.c_str(),"w"); //QOpen(fileName,"w");
        WriteMoldenATOMS_Repositioned(fp);
        WriteMoldenGTO(fp);
        vec dTs(n_mo); dTs.zeros();
        dTs(0)=99.0;
        WriteMoldenDiffDen(Cu.memptr(),Cu.memptr(),dTs.memptr(),dTs.memptr(),Cu.memptr(),Cu.memptr(),
                           the_scf->NOcc,the_scf->NOrb-the_scf->NOcc,the_scf->NOcc,the_scf->NOrb-the_scf->NOcc,fp);
        fclose(fp);
        
        if (ApplyImpulse) // Generate a power spectrum of the impulse for reference.
        {
            mat imp(10000,2);
            for (int i=0; i<10000; i+=1)
            {
                imp(i,0)=dt*(double)i;
                imp(i,1)=Mus->ImpulseAmp(dt*(double)i);
            }
            mat four=Fourier(imp,FourierZoom);
            four.save(logprefix+"FourierImpulse"+".csv", raw_ascii);
        }
        
    }
    
    void SetmanDensity()
    {
        int n_vir = n_mo - n_occ;
        int StartState = max(rem_read(REM_RTPPSTATE),0);
        cout << "Initializing from Linear Response State (starts from 1):" << StartState << endl;
        
        // To read out of the files written by setman.
        // Try Reading the RPA from setman.
        vec Xv(n_occ*n_vir),Yv(n_occ*n_vir);
        Xv.zeros(); Yv.zeros();
        
        bool RPA = true;
        
        FileMan(FM_READ,FILE_SET_TMP_RPA_X,FM_DP,n_vir*n_occ,0+n_vir*n_occ*(StartFromSetman-1),FM_BEG,Xv.memptr());
        if(RPA)
            FileMan(FM_READ,FILE_SET_TMP_RPA_Y,FM_DP,n_vir*n_occ,0+n_vir*n_occ*(StartFromSetman-1),FM_BEG,Yv.memptr());
        
        mat X(n_occ,n_vir),Y(n_occ,n_vir);
        cx_mat RhoRPA(n_mo,n_mo); RhoRPA.zeros();
        
        for (int a=0; a<n_vir; ++a)
            for (int i=0; i<n_occ; ++i)
            {
                X(i,a) = Xv(i*n_vir + a);
                Y(i,a) = Yv(i*n_vir + a);
            }
        
        if (n_mo<12 or Print > 1)
        {
            X.print("X");
            Y.print("Y");
        }
        
        for (int i=0; i<n_occ; ++i)
        {
            for (int j=0; j<n_occ; ++j)
            {
                for (int a=0; a<n_vir; ++a)
                {
                    RhoRPA(i,j) -= X(i, a)*X(j, a);
                    RhoRPA(i,j) += Y(i, a)*Y(j, a);
                }
            }
        }
        
        for (int b=0; b<n_vir; ++b)
        {
            for (int a=0; a<n_vir; ++a)
            {
                for (int i=0; i<n_occ; ++i)
                {
                    RhoRPA(a + n_occ,b+n_occ) += X(i,a)*X(i,b);
                    RhoRPA(a + n_occ,b+n_occ) -= Y(i,a)*Y(i,b);
                }
            }
        }
        
        /*
         vec tvec = evecs_.col(StartState-1); // is an OxV mat.
         // Add electron and subtract hole densities from the initial density.
         for (int i=0; i<n_occ; ++i)
         {
         for (int j=0; j<n_occ; ++j)
         {
         for (int a=0; a<n_vir; ++a)
         {
         Rho(i,j) -= tvec(i*n_vir + a)*tvec(j*n_vir + a);
         }
         }
         }
         
         for (int b=0; b<n_vir; ++b)
         {
         for (int a=0; a<n_vir; ++a)
         {
         for (int i=0; i<n_occ; ++i)
         Rho(a + n_occ,b+n_occ) += tvec(i*n_vir + a)*tvec(i*n_vir + b);
         }
         }
         */
        
        // Add this to the orig density matrix in the active space.
        Rho += (params["ExcitedFraction"])*RhoRPA.submat(firstOrb,firstOrb,lastOrb,lastOrb);
        
        cout << "Constructed Particle-Hole Density" << endl;
        cout << "Tr(Rho)" << trace(Rho) << endl;
        cout << "Idempotency: " << accu(Rho-Rho*Rho) << endl;

        Rho.diag().st().print("Raw Populations:");

        cout << "Ensuring positivity." << endl;
        Posify(Rho,n_e);
        Rho.diag().st().print("Initial Populations:");
    
    }
    
    void ReadRestart()
    {
        if (!(FileExists((logprefix+"Rho_lastsave").c_str())))
        {
            cout << "Cannot Restart... missing data." << endl;
            Pol.resize(5000,npol_col);
            Pol.zeros(); // Polarization Energy, Trace, Gap, Entropy, homo-lumo Coherence.
        }
        else
        {
            try{
                cout << "Reading old outputs to restart..." << endl;
                // Read in the old Pol.
                Pol.load(logprefix+"Pol.csv",raw_ascii);
                int i=2;
                for (; i<Pol.n_rows; ++i)
                    if (Pol(i,0) == 0.0)
                        break;
                if (i == Pol.n_rows-1)
                    cout << "Trouble Reading Pol File... " << endl;
                t = Pol(i-1,0);
                loop=i*DipolesEvery;
                loop2=i;
                loop3=int(loop/SaveEvery);
                cout << " *** Disabling save of fock energies and populations. *** " << endl;
                SavePopulations = false;
                SaveFockEnergies = false;
                cout << "Reading Active space Rho and starting from there." << endl;
                f_diag.load(logprefix+"f_diag_lastsave");
                Rho.load(logprefix+"Rho_lastsave");
                V.load(logprefix+"V_lastsave");
                Mus->load(logprefix+"Mus_lastsave_");
                Vs.load(logprefix+"Vs_lastsave");
                if (UpdateFockMatrix)
                {
                    HJK.load(logprefix+"HJK_lastsave");
                    RhoM12.load(logprefix+"RhoM12_lastsave");
                }
            }
            catch(...)
            {
                cout << "Restart -------- FAILURE ------- " << endl;
                throw;
            }
        }
    }
    
	void SplitLiouvillian(const arma::cx_mat& oldrho, arma::cx_mat& newrho, const double& tnow, bool& IsOn) const
	{
		if (TCLOn)
        {
            newrho.zeros();
			MyTCL->ContractGammaRhoMThreads(newrho,oldrho);
		}
	}
    
	void Split_RK4_Step(const arma::vec& f_diag, const arma::cx_mat& oldrho, arma::cx_mat& newrho, const double tnow, const double dt, bool& IsOn) const
	{
        if (!is_finite(oldrho) or !is_finite(f_diag))
        {
            cout << "Split_RK4_Step passed Garbage" << endl;
            throw 1;
        }
        
		arma::cx_mat f_halfstep(newrho); f_halfstep.eye();
		arma::cx_mat RhoHalfStepped(newrho); RhoHalfStepped.zeros();
        
        cx_vec f_diagc(f_diag.n_elem); f_diagc.set_real(f_diag);
        arma::cx_mat F = diagmat(f_diagc);
		Mus->ApplyField(F,tnow,IsOn);
        // Exponential propagator for Mu.
        vec eigval;
        cx_mat Cu;
        eig_sym(eigval,Cu,F);
        cx_mat Ud = exp(eigval*std::complex<double>(0.0,-0.5)*dt);
        cx_mat U = Cu*diagmat(Ud)*Cu.t();
		RhoHalfStepped = U*oldrho*U.t();
        
        if (!is_finite(newrho))
        {
            cout << "Garbage output in RhoHalfStepped" << endl;
            F.print("F");
            oldrho.print("oldrho");
            throw 1;
        }
        
		arma::cx_mat k1(newrho),k2(newrho),k3(newrho),k4(newrho),v2(newrho),v3(newrho),v4(newrho);
		k1.zeros(); k2.zeros(); k3.zeros(); k4.zeros();
		v2.zeros(); v3.zeros(); v4.zeros();
		
		SplitLiouvillian( RhoHalfStepped, k1,tnow,IsOn);
		v2 = (dt/2.0) * k1;
		v2 += RhoHalfStepped;
		SplitLiouvillian( RhoHalfStepped, k2,tnow+(dt/2.0),IsOn);
		v3 = (dt/2.0) * k2;
		v3 += RhoHalfStepped;
		SplitLiouvillian( RhoHalfStepped, k3,tnow+(dt/2.0),IsOn);
		v4 = (dt) * k3;
		v4 += RhoHalfStepped;
		SplitLiouvillian( RhoHalfStepped, k4,tnow+dt,IsOn);
		newrho = RhoHalfStepped;
		newrho += dt*(1.0/6.0)*k1;
		newrho += dt*(2.0/6.0)*k2;
		newrho += dt*(2.0/6.0)*k3;
		newrho += dt*(1.0/6.0)*k4;
        newrho = U*newrho*U.t();
    }
    
	void Split_RK4_Step_MMUT(const arma::vec& eigval, const arma::cx_mat& Cu , const arma::cx_mat& oldrho, arma::cx_mat& newrho, const double tnow, const double dt, bool& IsOn) const
	{
        cx_mat Ud = exp(eigval*std::complex<double>(0.0,-0.5)*dt);
        cx_mat U = Cu*diagmat(Ud)*Cu.t();
		cx_mat RhoHalfStepped = U*oldrho*U.t();
        
		arma::cx_mat k1(newrho),k2(newrho),k3(newrho),k4(newrho),v2(newrho),v3(newrho),v4(newrho);
		k1.zeros(); k2.zeros(); k3.zeros(); k4.zeros();
		v2.zeros(); v3.zeros(); v4.zeros();
		
		SplitLiouvillian( RhoHalfStepped, k1,tnow,IsOn);
		v2 = (dt/2.0) * k1;
		v2 += RhoHalfStepped;
		SplitLiouvillian(  RhoHalfStepped, k2,tnow+(dt/2.0),IsOn);
		v3 = (dt/2.0) * k2;
		v3 += RhoHalfStepped;
		SplitLiouvillian(  RhoHalfStepped, k3,tnow+(dt/2.0),IsOn);
		v4 = (dt) * k3;
		v4 += RhoHalfStepped;
		SplitLiouvillian(  RhoHalfStepped, k4,tnow+dt,IsOn);
		newrho = RhoHalfStepped;
		newrho += dt*(1.0/6.0)*k1;
		newrho += dt*(2.0/6.0)*k2;
		newrho += dt*(2.0/6.0)*k3;
		newrho += dt*(1.0/6.0)*k4;
        newrho = U*newrho*U.t();
    }
	
    void MMUTStep(arma::vec& f_diag, arma::cx_mat& HJK, arma::cx_mat& V, arma::cx_mat& Rho_, arma::cx_mat& RhoM12_, const double tnow , const double dt,bool& IsOn)
    {
        cx_mat CPrime(Rho_);
        // Fock Rebuild Rho at this time.
        CPrime = UpdateLiouvillian(f_diag,HJK,Rho_); // This rotates Rho, we must also rotate RhoM12!!!
        RhoM12_ = CPrime.t()*RhoM12_*CPrime;
        if (Stabilize)
            Posify(RhoM12_, n_e);
        
        // Make the ingredients for U.
        arma::cx_mat F(Rho_); F.zeros(); F.set_real(diagmat(f_diag));
		Mus->ApplyField(F,tnow,IsOn);
        // Exponential propagator for Mu.
        vec eigval; cx_mat Cu;
        eig_sym(eigval,Cu,F);
        
        // Full step RhoM12 to make new RhoM12.
        cx_mat NewRhoM12(Rho_);
        cx_mat NewRho(Rho_);
        Split_RK4_Step_MMUT(eigval, Cu, RhoM12_, NewRhoM12, tnow, dt, IsOn);
        // Half step that to make the new Rho.
        Split_RK4_Step_MMUT(eigval, Cu, NewRhoM12, NewRho, tnow, dt/2.0, IsOn);
        Rho_ = 0.5*(NewRho+NewRho.t());
        RhoM12_ = 0.5*(NewRhoM12+NewRhoM12.t());
    }
    
    
	void InitializeLiouvillian(vec& f_diag_, cx_mat& V, cx_mat& mu_x_, cx_mat& mu_y_, cx_mat& mu_z_)
	{
		int n = the_scf->NOrb;
		cx_mat Rhob(n,n); Rhob.zeros();
		cout << "Initializing the Liouvillian for " << n << " orbitals " << endl;
		the_scf->UpdateLiouvillian(Scratch, f_diag_,V,mu_x_,mu_y_,mu_z_,Rhob,old_fock,old_p,false);
	}
	
	// arg is in current Eigenbasis and mapped to Lowdin.
	cx_mat FullRho(const cx_mat& arg) const
	{
        if (ActiveSpace)
        {
            // Enlarge the density into the whole x space.
            cx_mat tmp = (Vs*arg*Vs.t());
            if (Print)
                cout << "FullRho: " << trace(tmp)<< trace(P0cx+tmp) << endl;
            return (P0cx+tmp);
        }
        else
            return (V*arg*V.t());
	}
    
    // Sync State members with another TDSCF object.
    // Ie: after this routine the propagations are in the same state.
    // Obviously between the Active space, incremental builds, etc. the number of state
    // variables has gotten out of control, and needs to be objectified.
    void Sync(const TDSCF* other)
    {
        cout << "P: " << SerialNumber << " Syncing from " << other->SerialNumber << endl;
        Mus->Sync(other->Mus);
        firstOrb = other->firstOrb;
        lastOrb = other->lastOrb;
        f_diag = other->f_diag;
        C=other->C;
        HJK=other->HJK;
        P0=other->P0;
        f_diagb=other->f_diagb;
        V0=other->V0;
        V=other->V;
        Vs=other->Vs;
        Csub=other->Csub;
        P0cx=other->P0cx;
        P_proj=other->P_proj;
        Q_proj=other->Q_proj;
        Ps_proj=other->Ps_proj;
        Qs_proj=other->Qs_proj;
        sH=other->sH;
        sHJK=other->sHJK;
        old_fock=other->old_fock;
        old_p=other->old_p;
        Rho=other->Rho;
        NewRho=other->NewRho;
        RhoM12=other->RhoM12;
        Mus->InitializeExpectation(Rho);
        cout << "Sync'd Mean-Field Energy: " << MeanFieldEnergy(Rho,V,HJK) << endl;
    }
    
	// P_ is in the current fock eigenbasis and will be updated.
    // V (the lowdin to fock transformation) is updated.
    // Fock is in the AO basis and will be updated.
    // The matrix returned is the Rotation matrix into the new fock eigenbasis from the old fock eigenbasis.
	cx_mat UpdateLiouvillian(vec& f_diag_, cx_mat& HJK, cx_mat& P_, bool Rebuild=true)
	{
        cx_mat Cprime(P_);
		if (Print>3)
			P_.print("P_ in update.");
		int n = the_scf->NOrb;
		int n_e = the_scf->NOcc;
        
		cx_mat Rhob = FullRho(P_); // Makes an X-basis density
        
		if (Print>3)
			Rhob.print("Rhob");
        
		if (Rebuild)
		{
			arma::cx_mat tmp;
            if (!ActiveSpace)
                Cprime = the_scf->UpdateLiouvillian(Scratch, f_diag_,V,HJK,tmp,tmp,Rhob,old_fock,old_p,IncFock && !(n_fock%100==0));
			else
                Cprime = the_scf->UpdateLiouvillian_ActiveSpace(Scratch,f_diag_,V,Vs,HJK,Rhob,P0cx,P_proj,Q_proj,Ps_proj,Qs_proj,old_fock,old_p,IncFock && !(n_fock%100==0));
            
            if (Stabilize)
                Posify(Rhob,the_scf->NOcc);
            
            if (!ActiveSpace)
            {
                P_ = V.t()*Rhob*V;
                Vs=V;
            }
            else
            {
                // Rebuild overall V-matrix.
                P_ = Vs.t()*Rhob*Vs;
            }
            
            Csub=the_scf->X*Vs;
            
            // Update Dipole Matrices Into current fock basis.
            Mus->update(Csub);
            // Update TCL.
            if (TCLOn && MyTCL != NULL)
                MyTCL->update(Csub,f_diag_);
		}
        else if (Stabilize)
			Posify(Rhob,the_scf->NOcc);
        if (!is_finite(Rhob))
        {
            cout << "Posify returned garbage... " << endl;
            throw 1;
        }
        
        if (Print>3)
            P_.print("P_");
		
        return Cprime;
	}
    
	double VonNeumannEntropy(cx_mat& rho,bool print=false) const
	{
		vec eigval;
		cx_mat Cprime;
		if (!eig_sym(eigval, Cprime, rho, "dc"))
		{
			return 0.0;
		}
		if (print)
			eigval.st().print("Natural Occs");
		vec omp = 1.0-eigval;
		vec olnp = log(omp);
		vec lnp = log(eigval);
		for (int i=0; i<eigval.n_elem; ++i)
		{
			if (eigval(i)<pow(10.0,-12.0))
				lnp(i)=0.0;
			if (omp(i)<pow(10.0,-12.0))
				olnp(i)=0.0;
		}
		return -2.0*(dot(eigval,lnp)+dot(omp,olnp));
	}
	
	double MeanFieldEnergy(cx_mat& rho, cx_mat& V, cx_mat& HJK) const
	{
        cx_mat Plao;
        if (!ActiveSpace)
        {
            Plao = V*rho*V.t();
            arma::cx_mat Pao = the_scf->X.t()*Plao*the_scf->X;
            double Etot = real(dot(Pao,the_scf->H)+ dot(Pao,HJK))+ the_scf->nuclear_energy()+the_scf->Ec+the_scf->Ex;
            return Etot;
        }
        else
        {
            Plao = P0cx+Vs*rho*Vs.t(); // P0cx is in the x basis.
            arma::cx_mat Pao = the_scf->X.t()*Plao*the_scf->X;
            double Etot = real(dot(Pao,the_scf->H)+ dot(Pao,HJK))+ the_scf->nuclear_energy()+the_scf->Ec+the_scf->Ex;
            return Etot;
        }
	}
	
	// Ie: with Aufbau occupations.
	double HundEnergy(cx_mat& rho, cx_mat& V, cx_mat& HJK) const
	{
		arma::cx_mat PTemp(the_scf->NOrb,the_scf->NOrb);
		PTemp.eye();
		PTemp.submat(the_scf->NOcc,the_scf->NOcc,the_scf->NOrb-1,the_scf->NOrb-1) *= 0.0;
		arma::cx_mat Plao = V*PTemp*(V.t());
		arma::cx_mat Pao = the_scf->X.t()*Plao*the_scf->X;
		double Etot = real(dot(Pao,the_scf->H)+ dot(Pao,HJK))+ the_scf->nuclear_energy()+the_scf->Ec+the_scf->Ex;
		return Etot;
	}
	
    // The other version of this routine can leave atoms too far away from the origin to actually be plotted. ;(
    // This version cleans that up.
    void WriteMoldenATOMS_Repositioned(FILE *fp) const
    {
        //
        // Write the [Atoms] section containing the geometry.
        //
        if (!fp) fp = stdout;
        
        // this will print ALL atoms (QM + MM) ... WAIT FUCK THAT...  Gotta undo that bit.
        int NAtoms = rem_read(REM_NATOMS);
        
        int *AtNo;
        double *Carts3,*xyz;
        get_carts(NULL,&Carts3,&AtNo,NULL);
        xyz = QAllocDouble(3*NAtoms);
        VRcopy(xyz,Carts3,3*NAtoms);
        
        bool useBohr = (rem_read(REM_INPUT_BOHR) == 1) ? true : false;
        if (!useBohr) VRscale(xyz,3*NAtoms,ConvFac(BOHRS_TO_ANGSTROMS));
        
        fprintf(fp,"[Molden Format]\n[Atoms] ");
        if (useBohr)
            fprintf(fp,"(AU)\n");
        else
            fprintf(fp,"(Angs)\n");
        
        double xm=0;
        double ym=0;
        double zm=0;
        for (int i = 0; i < NAtoms; i++){
            xm += xyz[3*i];
            ym += xyz[3*i+1];
            zm += xyz[3*i+2];
        }
        
        for (int i = 0; i < NAtoms; i++){
            String AtSymb = AtomicSymbol(AtNo[i]);
            char *atsymb = (char*)AtSymb;
            fprintf(fp,"%3s %6d %4d %15.8f %15.8f %15.8f\n",atsymb,
                    i+1,AtNo[i],xyz[3*i]-xm/NAtoms,xyz[3*i+1]-ym/NAtoms,xyz[3*i+2]-zm/NAtoms);
        }
        QFree(xyz);
    }
    
	// Allows for variable occupation numbers which are correctly plotted in GABEDIT.
	void WriteMoldenDiffDen(double *jCA, double *jCB, double *jEA, double *jEB, double *occa, double* occb,
							int NOccA,    int NVirA,    int NOccB,    int NVirB, FILE *fp, bool Rest=false) const
	{
		int NOccAPrt=NOccA;
		int NOccBPrt=NOccA;
		int NVirAPrt=NVirA;
		int NVirBPrt=NVirA;
		if (Rest)
			NOccBPrt = NVirBPrt = 0;
		//
		// Write the [MO] section using a "general" (i.e., user-specified)
		// set of MO coefficients.
		//
		// fp = destination file (opened by the caller).  Defaults
		//      to NULL for stdout.
		//
		// jCA,jCB = alpha/beta MO coefficients
		// jEA,jEB = alpha/beta orbital energies
		//
		// NOccA,NVirA,NOccB,NVirB
		//       = no. of occupied/virtual alpha/beta orbitals that are
		//         contained in the input jC.
		//
		// NOccAPrt = no. alpha occupieds to output (HOMO, HOMO-1, HOMO-2, ...)
		// NVirAPrt = no. alpha virtuals to output (LUMO, LUMO+1, LUMO+2, ...)
		// NOccBPrt, NVirBPrt = no. beta occupieds/virtuals to output
		//
		if (!fp) fp = stdout;
		int NBasis = bSetMgr.crntShlsStats(STAT_NBASIS);
		int NBas6D = bSetMgr.crntShlsStats(STAT_NBAS6D);
		int NAlpha = rem_read(REM_NALPHA);
		int NBeta  = rem_read(REM_NBETA);
		
		BasisSet basis(DEF_ID);
		LOGICAL *pureL = basis.deciPC();
		bool pureD = pureL[2],
		pureF = pureL[3],
		pureG = pureL[4];
		
		int NAtoms;
		if (rem_read(REM_QM_MM_INTERFACE) <= 0)
			NAtoms = rem_read(REM_NATOMS);
		else
			NAtoms = rem_read(REM_NATOMS_MODEL);
		
		int *AtNo,LMin,LMax,NShells,nfunc;
		int bCode = basis.code();
		get_carts(NULL,NULL,&AtNo,NULL);
		BasisID* BID = new BasisID [NAtoms];
		for (INTEGER i = 0; i < NAtoms; ++i)
			BID[i] = BasisID(AtNo[i],i+1,bCode);
		MolecularBasis molBasis(BID,NAtoms);
		delete [] BID;
		
		// Check that there are no basis functions with L > 3.
		// Molden supports L=4, but I don't think it's working yet here.
		
		int LMaxOverall=0;  // LMax for the entire basis
		for (int iAtom=0; iAtom < NAtoms; iAtom++){
			NShells = molBasis.pAtomicBasis(iAtom)->NumShells();
			for (int i=0; i < NShells; i++){
				LMax = molBasis.pAtomicBasis(iAtom)->LMax(i);
				if (LMax > 4){
					cout << "Molden output is not available for angular "
					<< "momentum beyond G\n";
					QWarn("Skipping Molden MO output");
					return;
				}
				else if (LMax > LMaxOverall)
					LMaxOverall = LMax;
			}
		}
		
		fprintf(fp,"[MO]\n");
		
		int *index = QAllocINTEGER(NBasis);
		getIndicesForMoldenMOs(index);
		
		if (NOccAPrt > 0){
			if (NOccAPrt > NOccA) NOccAPrt=NOccA; // that's all we have
			for (int i=NOccA-NOccAPrt; i < NOccA; i++){
				fprintf(fp,"Sym=X\nEne= %.6f\n",jEA[i]);
				fprintf(fp,"Spin=Alpha\nOccup=%.6f\n",occa[i]);
				for (int mu=0; mu < NBasis; mu++)
					fprintf(fp,"%6d%22.12e\n",mu+1,jCA[i*NBasis+index[mu]]);
			}
		}
		if (NVirAPrt > 0){
			if (NVirAPrt > NVirA) NVirAPrt=NVirA;
			for (int i=NOccA; i < NOccA+NVirAPrt; i++){
				fprintf(fp,"Sym=X\nEne= %.6f\n",jEA[i]);
				fprintf(fp,"Spin=Alpha\nOccup=%.6f\n",occa[i]);
				for (int mu=0; mu < NBasis; mu++)
					fprintf(fp,"%6d%22.12e\n",mu+1,jCA[i*NBasis+index[mu]]);
			}
		}
		if (NOccBPrt > 0){
			if (NOccBPrt > NOccB) NOccBPrt=NOccB; // that's all we have
			for (int i=NOccB-NOccBPrt; i < NOccB; i++){
				fprintf(fp,"Sym=X\nEne= %.6f\n",jEB[i]);
				fprintf(fp,"Spin=Beta \nOccup=%.6f\n",occb[i]);
				for (int mu=0; mu < NBasis; mu++)
					fprintf(fp,"%6d%22.12e\n",mu+1,jCB[i*NBasis+index[mu]]);
			}
		}
		if (NVirBPrt > 0){
			if (NVirBPrt > NVirB) NVirBPrt=NVirB;
			for (int i=NOccB; i < NOccB+NVirBPrt; i++){
				fprintf(fp,"Sym=X\nEne= %.6f\n",jEB[i]);
				fprintf(fp,"Spin=Beta \nOccup=%.6f\n",occb[i]);
				for (int mu=0; mu < NBasis; mu++)
					fprintf(fp,"%6d%22.12e\n",mu+1,jCB[i*NBasis+index[mu]]);
			}
		}
		
		// Molden assumes cartesian gaussians by default, so set an
		// appropriate flag in the case of pure (spherical) gaussians
		
		String purecart;
		bool needFlag=false;
		if (LMaxOverall == 2 && pureD){
			needFlag=true;
			purecart = "[5D]";
		}
		else if (LMaxOverall == 3){
			if (pureD || pureF) needFlag = true;
			
			if (pureD && pureF)
				purecart = "[5D7F]";
			else if (pureD && !pureF)
				purecart = "[5D10F]";
			else if (!pureD && pureF)
				purecart = "[7F]";
		}
		if (needFlag) fprintf(fp,"%s\n",(char*)purecart);
		QFree(index);
	}
	
    // the following routine is okay for gabedit (but only puts one density in the file.)
	void WriteEH(const cx_mat& rho,const mat& P0,const cx_mat& V, double t, int& Dest) const
	{
		int N = the_scf->NOrb;
		mat eigv(N,N);
		vec eige(N);
		// Generate the AO density.
		arma::cx_mat Plao = FullRho(rho);
		arma::mat Pao = real(the_scf->X.t()*Plao*the_scf->X);
        
		//Check trace:
		cout << "Writing Density Trace: " << trace(Pao*the_scf->AOS) ;
		Pao = Pao - P0;
		cout << " Difference Trace: " << trace(Pao*the_scf->AOS) << endl;
		// Now only the real part of this density can be responsible for n(r) generate the lowdin orbitals of that and write them to the file.
        
		eig_sym(eige, eigv, Pao*the_scf->AOS);
		eige.st().print("Difference Noccs:");
        // Occasionally get nan underflows...
        for (int i=0;i<the_scf->NOrb; ++i)
            for (int j=0;j<the_scf->NOrb; ++j)
                if (!is_finite(eigv(i,j)))
                    eigv(i,j)=0.0;
		eigv = eigv.t(); // now has dimensions MOxAO
		int h = the_scf->NOcc-1; int l = h+1;
		int nes=0; int nhs=0;
		vec ho(N), eo(N), e2o(N);
		mat hc(N,N); mat ec(N,N);
		ho.zeros(); eo.zeros();
		hc.zeros(); ec.zeros();
		for (int i=0;i<the_scf->NOrb; ++i)
		{
            if (!is_finite(eige(i)))
                continue;
			if (eige(i)<0.0)
			{
				ho(i) = eige(i);
				hc.row(i) += eigv.row(i);
			}
			else
			{
				eo(i) = eige(i);
				ec.row(i) += eigv.row(i);
			}
		}
		char str[12];
		sprintf(str, "%i", (int)Dest);
		std::string name(str);
		cout << "|electron| " << sum(eo) << " |hole| " << sum(ho) << endl;
        {
            String fileName  = String("logs/")+String(name.c_str())+String("er.molden");
            FILE *fp = fopen(fileName.c_str(),"w"); //QOpen(fileName,"w");
            WriteMoldenATOMS_Repositioned(fp);
            WriteMoldenGTO(fp);
            ec = ec.t();
            vec dTs(N); dTs.zeros();
            dTs(0)=t;
            WriteMoldenDiffDen(ec.memptr(),ec.memptr(),dTs.memptr(),dTs.memptr(),eo.memptr(),eo.memptr(),
                               the_scf->NOcc,the_scf->NOrb-the_scf->NOcc,the_scf->NOcc,the_scf->NOrb-the_scf->NOcc,fp);
            fclose(fp);
        }
        {
            String fileName  = String("logs/")+String(name.c_str())+String("hr.molden");
            FILE *fp = fopen(fileName.c_str(),"w"); //QOpen(fileName,"w");
            WriteMoldenATOMS_Repositioned(fp);
            WriteMoldenGTO(fp);
            hc = hc.t();
            ho *= -1.0;
            vec dTs(N); dTs.zeros();
            dTs(0)=t;
            WriteMoldenDiffDen(hc.memptr(),hc.memptr(),dTs.memptr(),dTs.memptr(),ho.memptr(),ho.memptr(),
                               the_scf->NOcc,the_scf->NOrb-the_scf->NOcc,the_scf->NOcc,the_scf->NOrb-the_scf->NOcc,fp);
            fclose(fp);
        }
		Dest++;
	}
	
    // Ridiculous implmentations of HF-energy and RIMP2 to debug
    // and ensure the integrals are correct
   /* void DebugRI(void)
    {
        // For now this routine calculates the MP2 CE.. just to debug.
        
        int LenV = megtot();
        int IBASIS = rem_read(REM_IBASIS);
        int Job,NBasAux,IBasisAux;
        int stat = STAT_NBASIS;
        IBasisAux = rem_read(REM_AUX_BASIS);
        ftnshlsstats(&NBasAux,&stat, &IBasisAux);
        double* qas=qalloc_start();

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
        invsqrtpq(&NBasAux, &Job);
        threading_policy::pop();
        
        // read (P|Q)^-1/2
        FileMan(FM_OPEN_RW, FILE_2C2E_INVSQ_INTS, 0, 0, 0, 0, 0);
        double *jPQ  = QAllocDouble(NBasAux * NBasAux);
        LongFileMan(FM_READ, FILE_2C2E_INVSQ_INTS, FM_DP, NBasAux*NBasAux, 0, FM_BEG, jPQ, 0);
        FileMan(FM_CLOSE, FILE_2C2E_INVSQ_INTS, 0, 0, 0, 0, 0);
        
        INTEGER ip[2], iq[2], iFile;
        if (true)
        {
            //  Form (ia|P) in order [a,P,i]
            iq[0] = 1; iq[1] = n_occ ; //q indicates occupied orbitals  0 -> NAlpha; // Actually I think these index from 1...
            ip[0] = 1; ip[1] = n_occ ;  //p indicates virtual orbitals  NAlpha+1 -> N;
            int p_len = ip[1]-ip[0]+1;
            int q_len = iq[1]-iq[0]+1;
            
            cout << "p_len" << p_len << endl;
            cout << "q_len" << q_len << endl;
            
            iFile = FILE_3C2Ea_INTS;
            threading_policy::enable_omp_only();
            // Also it's making the first q-batch of Bpq 0... which makes no sense.
            formpqr(&iFile, iq, ip, qas, &LenV); // For some reason this is dying if called AFTER the initial updateliouvillian.
            threading_policy::pop();
            
            //  Form B_iPa = sum_Q  I(i,Q,a) (P|Q)^-(1/2)
            int fileNum = FILE_BIJQa_INTS;
            double *BpP_q = QAllocDouble(NBasAux*p_len); // this is a misnomer... it's only BpP for a given q.
            double *IpP_q = QAllocDouble(NBasAux*p_len);
            
            cout << "NAux:" << NBasAux << endl;
            double *B  = QAllocDoubleWithInit(NBasAux*p_len*q_len);
            int as = q_len*NBasAux; // i stride.
            int rs = q_len; // R stride.
            
            for (int q=0; q<q_len; q++) {
                int Off8 = q * p_len * NBasAux;
                LongFileMan(FM_READ, FILE_3C2Ea_INTS, FM_DP, p_len*NBasAux, Off8, FM_BEG, IpP_q, 0);
                for (int a=0; a<p_len; ++a)
                    for (int R=0; R<NBasAux; ++R)
                        for (int S=0; S<NBasAux; ++S)
                            B[a*as+R*rs+q] += IpP_q[S*p_len+a]*jPQ[R*NBasAux+S];
                //LongFileMan(FM_WRITE, fileNum, FM_DP, p_len*NBasAux, Off8, FM_BEG, BpP_q, 0);
            };

            mat C = the_scf->C;
            mat Ct = the_scf->C.t();
            mat h = (Ct)*(the_scf->H)*C;
            double Ehf=0.0;
            double Eone=0.0;
            double EJ=0.0;
            double EK=0.0;
            for (int i=0; i<n_occ; ++i)
            {
                Ehf += 2.0*h(i,i);
                Eone += 2.0*h(i,i);
                for (int j=0; j<n_occ; ++j)
                {
                    for (int R=0; R<NBasAux; ++R)
                    {
                        Ehf += 2.0*B[i*as + R*rs + i]*B[j*as + R*rs + j];
                        EJ += 2.0*B[i*as + R*rs + i]*B[j*as + R*rs + j];
                        Ehf -= B[i*as + R*rs + j]*B[i*as + R*rs + j];
                        EK -= B[i*as + R*rs + j]*B[i*as + R*rs + j];
                    }
                }
            }
            cout << std::setprecision(9) << endl;
            cout << "HF Energy" << Ehf << endl;
            cout << "one Energy" << Eone << endl;
            cout << "J Energy" << EJ << endl;
            cout << "K Energy" << EK << endl;
            QFree(BpP_q);
            QFree(IpP_q);
            FileMan(FM_CLOSE, FILE_3C2Ea_INTS, 0, 0, 0, 0, 0);
        }
        if (true)
        {
            //  Form (ia|P) in order [a,P,i]
            iq[0] = 1; iq[1] = n_occ ; //q indicates occupied orbitals  0 -> NAlpha; // Actually I think these index from 1...
            ip[0] = n_occ+1;  ip[1] = n_mo ;  //p indicates virtual orbitals  NAlpha+1 -> N;
            int p_len = ip[1]-ip[0]+1;
            int q_len = iq[1]-iq[0]+1;
            
            cout << "p_len" << p_len << endl;
            cout << "q_len" << q_len << endl;
            
            iFile = FILE_3C2Ea_INTS;
            threading_policy::enable_omp_only();
            // Also it's making the first q-batch of Bpq 0... which makes no sense.
            formpqr(&iFile, iq, ip, qas, &LenV); // For some reason this is dying if called AFTER the initial updateliouvillian.
            threading_policy::pop();
            
            //  Form B_iPa = sum_Q  I(i,Q,a) (P|Q)^-(1/2)
            int fileNum = FILE_BIJQa_INTS;
            double *BpP_q = QAllocDouble(NBasAux*p_len); // this is a misnomer... it's only BpP for a given q.
            double *IpP_q = QAllocDouble(NBasAux*p_len);
            
            cout << "NAux:" << NBasAux << endl;
            double *B  = QAllocDoubleWithInit(NBasAux*p_len*q_len);
            int as = q_len*NBasAux; // i stride.
            int rs = q_len; // R stride.
            
            for (int q=0; q<q_len; q++) {
                int Off8 = q * p_len * NBasAux;
                LongFileMan(FM_READ, FILE_3C2Ea_INTS, FM_DP, p_len*NBasAux, Off8, FM_BEG, IpP_q, 0);
                for (int a=0; a<p_len; ++a)
                    for (int R=0; R<NBasAux; ++R)
                        for (int S=0; S<NBasAux; ++S)
                            B[a*as+R*rs+q] += IpP_q[S*p_len+a]*jPQ[R*NBasAux+S];
            };
            
            
            double ECorr = 0;
            int n_virt = n_mo - n_occ;
            double* f_diag = new double[n_mo];
            FileMan(FM_READ, FILE_MO_COEFS, FM_DP, n_mo, n*n_mo*2 , FM_BEG, f_diag, 0);
            for (int i=0; i<n_occ; ++i)
            {
                cout << "f_diag" << f_diag[i] << endl;
                for (int j=0; j<n_occ; ++j)
                {
                    for (int a=0; a<n_virt; ++a)
                    {
                        for (int b=0; b<n_virt; ++b)
                        {
                            double d = (f_diag[a+n_occ]+f_diag[b+n_occ]-f_diag[i]-f_diag[j]);
                            // (ia|jb)( 2*(ia|jb) - (ib|ja) ) / ( d )
                            for (int R=0; R<NBasAux; ++R)
                                for (int S=0; S<NBasAux; ++S)
                                    ECorr += B[a*as+R*rs+i]*B[b*as+R*rs+j]*(2.0*B[a*as+S*rs+i]*B[b*as+S*rs+j] - B[b*as+S*rs+i]*B[a*as+S*rs+j] )/d;
                        }
                    }
                }
            }
            cout << "Correlation Energy: " << std::setprecision(6) << ECorr << endl;
            QFree(BpP_q);
            QFree(IpP_q);
            QFree(jPQ);
            FileMan(FM_CLOSE, FILE_3C2Ea_INTS, 0, 0, 0, 0, 0);
        }
        
    }
    */
    
	void Posify(cx_mat& rho,int N) const
	{
		int n = rho.n_rows;
		if (!is_finite(rho))
		{
            cout << " Posify passed garbage" << endl;
            throw 1;
		}
		vec eigval;
		cx_mat u;
		cx_mat Rhob(rho); Rhob.zeros();
		Rhob.eye();
		Rhob.submat(N,N,rho.n_cols-1,rho.n_cols-1) *= 0.0;
		//rho.diag().st().print("input rho");
		rho -= Rhob; // Difference density.
		rho = 0.5*(rho+rho.t());
		if (abs(trace(rho)) < pow(10.0,-13.0))
		{
			rho+=Rhob;
			if (Print)
				cout << " too small, not stabilizing trace " << endl;
		}
		else
		{
			// Trace and e-h symmetry correction.
			if (!eig_sym(eigval, u, rho, "std"))
			{
				rho.print("rho-eig sym failed.");
                throw 1;
			}
			double ne = 0.0; double nh=0.0;
			vec IsHole(rho.n_cols); IsHole.zeros();
			vec IsE(rho.n_cols); IsE.zeros();
			for (int i=0; i<rho.n_cols;++i)
			{
				if (abs(eigval(i))>1.0)
				{
					eigval(i) = ((0 < eigval(i)) - (eigval(i) < 0))*1.0;
				}
				if (eigval(i)<0) // it's a hole
				{
					IsHole(i) = 1.0;
					nh+=eigval(i);
				}
				else
				{
					IsE(i) = 1.0;
					ne+=eigval(i);
				}
			}
			if (abs(sum(eigval%IsHole)) < abs(sum(eigval%IsE)))
			{
				// fewer holes than electrons, scale the electrons down so that they match
                if ( abs(sum(eigval%IsE)) != 0.0)
                {
                    IsE *= abs(sum(eigval%IsHole))/abs(sum(eigval%IsE));
                    eigval = (eigval%(IsE+IsHole));
                }
			}
			else
			{
				// Fewer electrons than holes, scale the holes down.
                if ( abs(sum(sum(eigval%IsHole))) != 0.0)
                {
                    IsHole *= abs(sum(eigval%IsE))/abs(sum(eigval%IsHole));
                    eigval = (eigval%(IsE+IsHole));
                }
			}
			if (Print)
				cout << " Difference Corrected to: " << sum(eigval);
			rho=u*diagmat(eigval)*u.t()+Rhob;
			if (Print)
				cout << " Trace Corrected to: " << trace(rho) << endl;
		}
		// positivity correction.
		if (Stabilize > 1)
		{
			vec eigval;
			cx_mat u;
			if (!eig_sym(eigval, u, rho, "std"))
			{
				cout << "eig_sym failed" << endl;
				rho.print("rho");
				throw 1;
			}
			if (Print)
				eigval.st().print("Nat Occs. before"); // In ascending order.
			for (int i=0; i<rho.n_cols;++i)
			{
				if (eigval(i)<0)
					eigval(i)*=0.0;
				else if (eigval(i)>1.0)
					eigval(i)=1.0;
			}
            double missing = (double)N-sum(eigval);
            if (abs(missing) < pow(10.0,-10.0))
                return;
			if (n-N-1<0)
			{
				rho=u*diagmat(eigval)*u.t();
				return;
			}
			if (Print)
				cout << "Positivity Error:" << missing;
			if (missing < 0.0 && eigval(n-N-1)+missing > 0.0)
				eigval(n-N-1) += missing; // Subtract from lumo-like
			else if (missing < 0.0)
				eigval(n-N) += missing;  // Subtract from homo-like
			else if (missing > 0.0 && eigval(n-N)+missing < 1.0)
				eigval(n-N) += missing; //  add to homo-like
			else
				eigval(n-N-1) += missing; //  add to lumo-like
			if (Print)
				eigval.st().print("Nat Occs. after"); // In ascending order.
			missing = (double)N-sum(eigval);
//			if (Print)
//				cout << " corrected to: " << missing << endl;
			rho=u*diagmat(eigval)*u.t();
// Finally recheck the diagonal is positive.
            for (int i=0; i<n;++i)
            {
                if (real(rho(i,i))<0.0)
                    rho(i,i)=0.0;
            }
            
		}
		return;
	}
    
    // Do a shifted/scaled fourier transform of polarization in each direction.
    mat Fourier(const mat& pol, double zoom = 1.0/20.0, int axis=0, bool dering=true) const
    {
        cout << "Computing Fourier Transform ..." << endl;
        mat tore;
        double dt = pol(3,0)-pol(2,0);
        {
            int ndata = (pol.n_rows%2==0)? pol.n_rows : pol.n_rows-1; // Keep an even amount of data
            for (int r=1; r<ndata; ++r)
                if (pol(r,0)==0.0)
                    ndata =	(r%2==0)?r:r-1;  // Ignore trailing zeros.
            mat data(ndata,1);
            data.col(0) = pol.rows(0,ndata-1).col(1+axis);
            data = data - mean(data); // subtract any zero frequency component.

            if (dering)
            {
                double gam = log(0.00001/(ndata*ndata));
                mat de = linspace<mat>(0.0,ndata,ndata);
                de = exp(-1.0*de%de*gam);
                data = data%de;
            }

            cx_mat f(ndata,1); f.zeros();
            mat fr(ndata/2,1), fi(ndata/2,1); fr.zeros(); fi.zeros();

#pragma omp parallel for schedule(guided)
            for (int r=0; r<ndata; ++r)
                f(r,0) = exp(std::complex<double>(0.0,2.0*M_PI)*zoom*(r-1.0)/ndata);

#pragma omp parallel for schedule(guided)
            for (int i=0; i<ndata/2; ++i) // I only even generate the positive frequencies.
            {
                cx_mat ftmp = pow(f,std::complex<double>((double)i,0.0));
                fr(i,0) = real(accu(ftmp%data));
                fi(i,0) = imag(accu(ftmp%data));
            }

            tore.resize(ndata/2,3);
            for (int i=0; i<ndata/2; ++i)
                tore(i,0) = M_PI*(27.2113*DipolesEvery/dt)*zoom*i/ndata;
            tore.col(1) = -1.0*fi.col(0);
            tore.col(2) = fr.col(0);
        }
        return tore;
    }
    
    TDSCF(rks* the_scf_,const std::map<std::string,double> Params_ = std::map<std::string,double>(), TDSCF_Temporaries* Scratch_ = NULL) : the_scf(the_scf_), logprefix("./logs/"), n_ao(the_scf_->N),n_mo(the_scf_->NOrb), n_occ(the_scf_->NOcc), n_e(the_scf_->NOcc), MyTCL(NULL), npol_col(10), ee2(NULL)
	{
        if (Scratch_!=NULL)
            Scratch = Scratch_;
        else
            Scratch = new TDSCF_Temporaries();
        
		cout << endl;
		cout << "===================================" << endl;
		cout << "|  Realtime TDSCF module          |" << endl;
		cout << "===================================" << endl;
		cout << "| J. Parkhill, T. Nguyen          |" << endl;
		cout << "| J. Koh, J. Herr,  K. Yao        |" << endl;
		cout << "===================================" << endl;
        cout << "| Refs: 10.1021/acs.jctc.5b00262  |" << endl;
        cout << "|       10.1063/1.4916822         |" << endl;
		cout << "===================================" << endl;
		n = n_mo; cout << "n_ao:" << n_ao << " n_mo " << n_mo << " n_e/2 " << n_occ << endl;
		n_fock = 0; // number of fock updates.
		      
		params = ReadParameters(Params_);
        if (!RunDynamics)
            return;
        
		V.resize(n,n); V.eye(); // The Change of basis between the Lowdin AO's and the current fock eigenbasis (in which everything is propagated)
		Vs.resize(n,n); Vs.eye(); // The Change of basis between the Lowdin AO's and the current fock eigenbasis (in which everything is propagated)
        
        C.resize(n,n); C.zeros(); // = X*V.
        HJK.resize(n,n); HJK.zeros(); // The Change of basis between the Lowdin AO's and the current fock eigenbasis (in which everything is propagated)
        
        P0=the_scf->P; // the initial density matrix (ao basis)
        f_diagb.resize(n_mo);
		cx_mat mu_xb(n_ao,n_ao), mu_yb(n_ao,n_ao), mu_zb(n_ao,n_ao); // Stored in the ao representation and transformed on demand.
		{
			cx_mat Rhob(n,n); Rhob.zeros();
			InitializeLiouvillian(f_diagb,V,mu_xb,mu_yb,mu_zb);
            f_diag = f_diagb;
			C = the_scf->X*V;
		}
		V0=V; // the initial mo-coeffs in terms of Lowdin (x-basis).
		
        // Note the mu_x,y etc. are provided to FieldMatrices in the X basis.)
        if (ApplyNoOscField)
            Mus = new StaticField(mu_xb, mu_yb, mu_zb, FieldVector, FieldAmplitude, Tau, tOn);
        else
            Mus = new OpticalField(mu_xb, mu_yb, mu_zb, FieldVector, ApplyImpulse, ApplyCw, FieldAmplitude, FieldFreq, Tau, tOn, ApplyImpulse2, FieldAmplitude2, FieldFreq2, Tau2, tOn2);
        
        // Plot the Basis' approximation to the x field if you so desire it.
#ifndef RELEASE
        PlotFields();
#endif
        
		// figure out which block of orbitals we'll propagate within.
		firstOrb = 0;
		lastOrb = n-1;
        f_diag.resize(n); f_diag.zeros();
        Csub = the_scf->X*V;
        
		if (ActiveSpace)
            SetupActiveSpace();
		
		cout << "Propagating space contains: " << n_e*2 << " singlet fermions in " << n << " orbitals "<< endl;
        
        Rho.resize(n,n); NewRho.resize(n,n);
        Rho.zeros(); NewRho.zeros();
		Rho.eye(); Rho.submat(n_e,n_e,n-1,n-1) *= 0.0;
		
        if (ActiveSpace)
        {
            UpdateLiouvillian( f_diag, HJK, Rho);
            sH = Vs.t()*(the_scf->X*(the_scf->H)*the_scf->X.t())*Vs;
            sHJK = Vs.t()*(the_scf->X*HJK*the_scf->X.t())*Vs;
            cx_mat Pt = the_scf->X.t()*P0cx*the_scf->X;
            ECore = real(dot(Pt,the_scf->H)+ dot(Pt,HJK));
            cout << std::setprecision(9) << "Mean field Energy of initial state (Active Space): " << MeanFieldEnergy(Rho,V,HJK) << endl;
        }
        else
        {
            f_diag = f_diagb;
            cout << std::setprecision(9) << "Mean field Energy of initial state: " << MeanFieldEnergy(Rho,V,HJK) << endl;
        }
        cout << "Koopman's Excitation estimate: " << f_diag(n_e)-f_diag(n_e-1) << " (eH) " <<  27.2113*(f_diag(n_e)-f_diag(n_e-1)) << " (eV) "<< endl;

        if (TCLOn && MyTCL == NULL)
            MyTCL = new TCLMatrices(n,n_ao,Csub,f_diag,n_e,params);
        
        if (StartFromSetman>0)
            SetmanDensity();
        else if (InitialCondition == 1)
        {
            cout << "WARNING -- WARNING" << endl;
            cout << "Initalizing into a particle hole excitation out of: " << n_e-1 << " and rebuilding the fock matrix (once)" << endl;
            Rho(n_e-1,n_e-1)=0.0;    Rho(n_e,n_e)=1.0; // initialize into a particle hole excitation
        }
        
        if (UpdateFockMatrix)
        {
            UpdateLiouvillian(f_diag, HJK, Rho); // update the eigenvalues and TCL.
        }
        
		C = the_scf->X*V;
		Csub=C.submat(0,firstOrb,n_ao-1,lastOrb);
		cout << "Finished Calculating the Initial Liouvillian, Energy: " << MeanFieldEnergy(Rho,V,HJK) << endl;
		f_diag.st().print("Initial Eigenvalues:");
		cout << "Excitation Gap: " << (f_diag(n_e)-f_diag(n_e-1)) << " Eh : " << (f_diag(n_e)-f_diag(n_e-1))*27.2113 << "eV" << endl;
		cout << "Condition of Fock Matrix: " << 1.0/(f_diag(n-1)-f_diag(0)) << " timestep (should be smaller) " << dt << endl;
		Rho.diag().st().print("Populations after first Fock build:");
        cout << "Idempotency: " << accu(Rho-Rho*Rho) <<  endl;
		Mus->update(Csub);
		cout << "Homo->Lumo transition strength: " << Mus->muxo(n_e-1,n_e) << endl;
        
		Mus->InitializeExpectation(Rho);
        t=0; loop=0; loop2=0; loop3=0;
        RhoM12=Rho;
        
        if (!Restart)
        {
            Pol.resize(5000,npol_col);
            Pol.zeros(); // Polarization Energy, Trace, Gap, Entropy, homo-lumo Coherence.
        }
        else
            ReadRestart();
        
		if (SavePopulations)
		{
			Pops = mat(100,n+1);
			Pops.zeros();
		}
		if (SaveFockEnergies)
		{
			FockEs = mat(100,n+1);
			FockEs.zeros();
		}
        
        if (Corr)
        {
            params["n_mo"] = n_mo;
            params["n_occ"] = n_occ;
			ee2 = new EE2(n_mo,n_occ, the_scf, Rho, V, params);
            Posify(Rho, n_e);
            // Print initial Rhodot.
            RhoM12=Rho;
            // Zero the dipoles again.
            C=the_scf->X*V;
            Mus->update(C);
            Mus->InitializeExpectation(Rho);
		}
        
        // If the change in the density is significant enough initiate a fock build...
        // These determine if that's satisfied.
        LastFockPopulations.resize(n);
        LastFockEnergies.resize(n);
        LastFockPopulations = Rho.diag();
        LastFockEnergies = f_diag;
        Entropy = VonNeumannEntropy(Rho,true);
        if ((UpdateMarkovRates or StartFromSetman) && TCLOn && MyTCL != NULL)
        {
            MyTCL->rateUpdate();
            cout << "Gap in units of KbT: " << (f_diag(n_e)-f_diag(n_e-1))*MyTCL->Beta  << endl;
        }
        
		cout << "=============================================" << endl;
        if (SerialNumber)
            cout << "Initalization of TDSCF SerialNumber " << SerialNumber << " complete" << endl;
        else
            cout << "Initalization of TDSCF complete" << endl;
		cout << "=============================================" << endl;
		wallt0 = time(0); // This is in wall-seconds.
        if (!rem_read(REM_RTPUMPPROBE))
            Propagate();
		return;
	}
    
    bool Propagate()
    {
        bool IsOn;
        double WallHrPerPs=0;
        if (Corr && !params["RIFock"])
            cout << "---Corr TDHF---" << endl;
        else if (UpdateFockMatrix)
            cout << "---MMU TDHF---" << endl;
        else
            cout << "---Frozen Orbitals---" << endl;
        
        Rho.diag().st().print("Initial Populations:");
        
        while(true)
        {
            // The Timestep ---------------------------------------------------------------
			if (Corr)
            {
		        Energy = ee2->step(the_scf, Mus, f_diag, V, Rho, RhoM12, t, dt, IsOn);
                if (Stabilize)
                {
                    Posify(Rho, n_e);
                    Posify(RhoM12, n_e);
                }
                n_fock++;
                // Is a rate Rebuild Called for?
                vec tmp = f_diag - LastFockEnergies;
                double NewRates = norm(tmp);
                if (NewRates > RateBuildThresh && UpdateMarkovRates && MyTCL != NULL)
                {
                    MyTCL->rateUpdate();
                    LastFockEnergies = f_diag;
                }
            }
            else if (UpdateFockMatrix)
            {
                MMUTStep(f_diag, HJK, V, Rho, RhoM12, t, dt,IsOn);
                n_fock++;
                // Is a rate Rebuild Called for?
                vec tmp = f_diag - LastFockEnergies;
                double NewRates = norm(tmp);
                if (NewRates > RateBuildThresh && UpdateMarkovRates && MyTCL != NULL)
                {
                    MyTCL->rateUpdate();
                    LastFockEnergies = f_diag;
                }
            }
            else
            {
                Split_RK4_Step(f_diag, Rho, NewRho, t, dt,IsOn);
                Rho=0.5*(NewRho+NewRho.t());
            }
            LastFockPopulations = Rho.diag();
            t+=dt;
            // Logging ---------------------------------------------------------------
            if (SaveFockEvery)
            {
                if (n_fock%SaveFockEvery == 0 && UpdateFockMatrix)
                {
                    int rw=n_fock/SaveFockEvery;
                    if (SaveFockEnergies)
                    {
                        FockEs(rw,0) = t;
                        if (FockEs.n_cols != f_diag.n_elem+1)
                            FockEs.resize(FockEs.n_rows,f_diag.n_elem+1);
                        FockEs.row(rw).cols(1,n) = real(f_diag.st());
                        if ( FockEs.n_rows-rw < 10)
                            FockEs.resize(FockEs.n_rows+2000,n+1);
                    }
                }
            }
            if (SavePopulations && (loop2%((int)params["SavePopulationsEvery"])==0))
            {
                if ((loop2%(int)params["SavePopulationsEvery"])==0)
                {
                int rw;
                for (rw=1;rw<Pops.n_rows; ++rw)
                    if (Pops(rw,0)==0)
                        break;
                Pops(rw,0)=t;
                if (params["PrntPopulationType"] == 0.0)
                {
                    cx_mat tmp;
                    if (ActiveSpace)
                        tmp=(Vs*Rho*Vs.t()) + P0cx;
                    else
                        tmp=(V*Rho*V.t());
                    cx_mat RhoIn0 = V0.t()*tmp*V0; // Orig. Basis density matrix.
                    if (Pops.n_cols != RhoIn0.n_cols+1)
                        Pops.resize(Pops.n_rows,RhoIn0.n_cols+1);
                    Pops.row(rw).cols(1,RhoIn0.n_cols) = real(RhoIn0.diag().t());
                    if ( Pops.n_rows-rw <10)
                        Pops.resize(Pops.n_rows+2000,RhoIn0.n_cols+1);
                }
                else if (params["PrntPopulationType"] == 1.0)
                {
                    vec nocs; cx_mat nos;
                    eig_sym(nocs, nos, Rho);
                    if (Pops.n_cols != nocs.n_cols+1)
                        Pops.resize(Pops.n_rows+2000,nocs.n_cols+1);
                    Pops.row(rw).cols(1,nocs.n_cols) = nocs.st();
                    if ( Pops.n_rows-rw <10)
                        Pops.resize(Pops.n_rows+2000,nocs.n_cols+1);
                }
                }
            }
            if (loop%DipolesEvery==0 && DipolesEvery)
            {
                Entropy = VonNeumannEntropy(Rho,Print>0);
                Energy = MeanFieldEnergy(Rho,V,HJK);
                vec tmp = Mus->Expectation(Rho);
                Pol(loop2,0) = t; // in atomic units.
                Pol(loop2,1) = tmp(0);
                Pol(loop2,2) = tmp(1);
                Pol(loop2,3) = tmp(2);
                Pol(loop2,4) = Energy;
                Pol(loop2,5) = real(trace(Rho));
                Pol(loop2,6) = (f_diag(n_e)-f_diag(n_e-1));
                Pol(loop2,7) = Entropy;
                Pol(loop2,8) = real(Rho(n_e-1,n_e));
                if (Corr)
                    Pol(loop2,9) = ee2->Ehf;
                if (Pol.n_rows-loop2<100)
                    Pol.resize(Pol.n_rows+20000,Pol.n_cols);
                loop2 +=1;
            }
            if (SaveEvery)
                if (loop%SaveEvery==0)
                {
                    real(Rho.diag().st()).print("Pops:");

                    if (SaveDipoles)
                    {
                        // Also write the density and natural orbitals so they can be visualized.
                        Pol.save(logprefix+"Pol"+".csv", raw_ascii);
                        if (SavePopulations)
                            Pops.save(logprefix+"Pops"+".csv", raw_ascii);
                        if (SaveFockEnergies)
                            FockEs.save(logprefix+"FockEigs"+".csv", raw_ascii);
                        // Save the density matrix for possible restart.
                        if (params["CanRestart"])
                        {
                            Rho.save(logprefix+"Rho_lastsave",arma_binary);
                            if (!ActiveSpace)
                                FullRho(Rho).save(logprefix+"Px_TDSCF",arma_binary);
                            if (loop < 10)
                            {
                                f_diag.save(logprefix+"f_diag_lastsave",arma_binary);
                                V.save(logprefix+"V_lastsave",arma_binary);
                                Vs.save(logprefix+"Vs_lastsave",arma_binary);
                                P0.save(logprefix+"P0_lastsave",arma_binary);
                                Mus->save(logprefix+"Mus_lastsave_");
                            }
                            if (UpdateFockMatrix)
                            {
                                f_diag.save(logprefix+"f_diag_lastsave",arma_binary);
                                V.save(logprefix+"V_lastsave",arma_binary);
                                Vs.save(logprefix+"Vs_lastsave",arma_binary);
                                HJK.save(logprefix+"HJK_lastsave");
                                RhoM12.save(logprefix+"RhoM12_lastsave");
                            }
                            // I have to fix the restart feature anyways.
                        }
                    }
                    if (WriteDensities > 0)
                        WriteEH(Rho, P0, V, t, loop3);
                }
            if(FourierEvery)
                if (loop%FourierEvery==0 && loop > 10)
                {
                    mat four_x=Fourier(Pol,FourierZoom,0);
                    four_x.save(logprefix+"Fourier_x"+".csv", raw_ascii);
                    mat four_y=Fourier(Pol,FourierZoom,1);
                    four_y.save(logprefix+"Fourier_y"+".csv", raw_ascii);
                    mat four_z=Fourier(Pol,FourierZoom,2);
                    four_z.save(logprefix+"Fourier_z"+".csv", raw_ascii);
                }
            
            // Finish Iteration ---------------------------------------------------------------
            WallHrPerPs = ((time(0)-wallt0)/(60.0*60.0))/(t*FsPerAu/1000.0);
            double Elapsed = (int)(time(0)-wallt0)/60.0;
            double Remaining = (((double)(max(MaxIter-loop, loop%(Stutter+1) )))*(Elapsed/loop));
            
            if(StatusEvery)
                if (loop%StatusEvery==0)
                {
                    cout << std::setprecision(6);
                    if (SerialNumber)
                        cout << "P: " << SerialNumber << " ITER: " << loop << " T: " << std::setprecision(2) << t*FsPerAu << "(fs) dt " << dt*FsPerAu  << "(fs) Hr/Ps: " << WallHrPerPs << " - Lpsd/Rem.: " << Elapsed << ", " << Remaining << " (min) Tr.Dev: " << abs(n_e-trace(Rho)) << " Hrm: " << abs(accu(Rho-Rho.t())) << std::setprecision(6) << " Enrgy: " << Energy << " Entr: " << Entropy <<  " Fld " << IsOn << " NFk: " << n_fock << endl;
                    else
                        cout << "ITER: " << loop << " T: " << std::setprecision(2) << t*FsPerAu << "(fs)  dt " << dt*FsPerAu  << "(fs) Hr/Ps: " << WallHrPerPs << " - Lpsd/Rem.: " << Elapsed << ", " << Remaining << " (min) Tr.Dev: " << abs(n_e-trace(Rho)) << " Hrm: " << abs(accu(Rho-Rho.t()))  << std::setprecision(6) << " Enrgy: " << Energy << " Entr: " << Entropy  << " Fld " << IsOn << " NFk: " << n_fock << endl;
                    if (Print)
                        Rho.diag().st().print("Populations");
                    Mus->Expectation(Rho).st().print("Mu");
                }
            
            loop +=1;
            
            if (MaxIter)
                if (loop>MaxIter)
                {
                    cout << "Exiting Because MaxIter exceeded. SerialNumber"<< SerialNumber << " Iter " << loop << " of " << MaxIter << endl;
                    Posify(Rho,n_e);
                    Posify(RhoM12,n_e);
                    return false;
                }
            if ((Stutter > 0) && (loop%Stutter==0) && loop>0)
            {
                Posify(Rho,n_e);
                Posify(RhoM12,n_e);
                return true;
            }
        }
    }
    
    ~TDSCF()
    {
        cout << "~TDSCF()" << endl;
        
        UpdateLiouvillian(f_diag,HJK,Rho);
		f_diag.st().print("Final Eigenvalues");
		Rho.diag().st().print("Final Populations");
		cout << "FINAL MEAN FIELD ENERGY: " << Energy << endl;
		cout << "FINAL ENTROPY:" << Entropy << endl;
		if (MyTCL!=NULL)
		{
			cout << "FINAL ENTROPY (hartrees):" << Entropy/(MyTCL->Beta) << endl;
			cout << "FINAL THERMODYNAMIC ENERGY:" << Energy-Entropy/(MyTCL->Beta) << endl;
            delete MyTCL;
            MyTCL=NULL;
		}
		cout << "FINAL HUNDS MEAN FIELD ENERGY: " << HundEnergy(Rho,V,HJK) << endl;
		// Save P-Matrix in Lowdin basis.
        FullRho(Rho).save(logprefix+"Px_TDSCF",arma_binary);
        if (Mus!=NULL)
        {
            delete Mus;
            Mus=NULL;
        }
		if (ee2 != NULL)
        {
			delete ee2;
            ee2=NULL;
        }
    }
    
    
    // This is for pump-probe.
    // Creates a special output files that contain:
    // Pumped dipole, Nonstationary dipole, Corrected dipole
    // and Fouriers of the above.
    void TransientAbsorptionCorrection(const TDSCF* NonSt, const TDSCF* GSA) const
    {
        cout << "Calculating Corrected TA Spectrum. It will go into path " << logprefix << endl;
        if (Pol.n_rows < loop2 or NonSt->Pol.n_rows < loop2)
        {
            cout << "Storage error " << loop2 << " : " << Pol.n_rows << " : " << NonSt->Pol.n_rows  << endl;
            throw;
        }
        
        // Store all three pol files for reference.
        Pol.save(logprefix+"Pol.csv",raw_ascii);
        NonSt->Pol.save(logprefix+"RefCorr.csv",raw_ascii);
        GSA->Pol.save(logprefix+"GS_Pol.csv",raw_ascii);
        
        mat PPol(loop2,4); // time , x1 y1 z1 ...
        PPol.zeros();
        PPol(arma::span(0,loop2-1), 0) = Pol(arma::span(0,loop2-1),0);
        PPol(arma::span(0,loop2-1), arma::span(1,3)) = Pol(arma::span(0,loop2-1), arma::span(1,3)) - NonSt->Pol(arma::span(0,loop2-1), arma::span(1,3)) - GSA->Pol(arma::span(0,loop2-1), arma::span(1,3));
        PPol.save(logprefix+"TAPol.csv",raw_ascii);

        mat gsa_x=Fourier(GSA->Pol,FourierZoom,0);
        gsa_x.save(logprefix+"Fourier_GS_x"+".csv", raw_ascii);
        mat gsa_y=Fourier(GSA->Pol,FourierZoom,1);
        gsa_y.save(logprefix+"Fourier_GS_y"+".csv", raw_ascii);
        mat gsa_z=Fourier(GSA->Pol,FourierZoom,2);
        gsa_y.save(logprefix+"Fourier_GS_z"+".csv", raw_ascii);
        
        mat four_x=Fourier(PPol,FourierZoom,0);
        four_x.save(logprefix+"Fourier_TA_x"+".csv", raw_ascii);
        mat four_y=Fourier(PPol,FourierZoom,1);
        four_y.save(logprefix+"Fourier_TA_y"+".csv", raw_ascii);
        mat four_z=Fourier(PPol,FourierZoom,2);
        four_z.save(logprefix+"Fourier_TA_z"+".csv", raw_ascii);
    }
    
};


// Pump-probe job requires 5 propagations.
// A) begin from a linear response excited state.
// B) Propagate that state without Fock build for t_delay fs.
// C) Allow the system to warm up for a bit with Fockupdates.
// D) Begin three 10fs propagations:
//  1 pump & probe, 2 - pump no probe (to correct nonstationarity) 3 - ground state (to get Delta A.)
//  Finally the dipole is made from 1-(2+3)
inline void PumpProbeJob(rks* pr)
{
    std::map<std::string,double> NoField, xField, Rlx, Gs, Warm;
    TDSCF_Temporaries* Scratch = new TDSCF_Temporaries();
    // First the relaxation job.
    Rlx["ApplyImpulse"] = 0;
    Rlx["StartFromSetman"] = int(rem_read(REM_RTPPSTATE));
    Rlx["UpdateFockMatrix"] = 0;
    Rlx["SerialNumber"] = 1;
    Rlx["Stabilize"] = 2;
    Rlx["Decoherence"] = 0;
    Rlx["SaveDipoles"] = 0;
    Rlx["DipolesEvery"] = 1000;
    Rlx["StatusEvery"] = 1000;
    Rlx["SaveEvery"] = 5000;
    Rlx["SavePopulations"] = 1;
    Rlx["FourierEvery"] = 0;
    Rlx["WriteDensities"] = 0;
    Rlx["SaveFockEvery"] = 0;
    Rlx["TCLOn"] = 1;
    Rlx["dt"] = 0.45; // Enormous steps can be taken because only the populations are being updated.
    Rlx["MaxIter"] = 0; // Perform an infinite number of propagations.

    Warm["TCLOn"] = 1;
    Warm["ApplyImpulse"] = 0;
    Warm["ApplyImpulse2"] = 0;
    Warm["ApplyCw"] = 0;
    Warm["StartFromSetman"] = 0;
    Warm["UpdateFockMatrix"] = 0;
    Warm["Decoherence"] = 1;
    Warm["SaveDipoles"] = 0;
    Warm["Stabilize"] = 2;
    Warm["SaveEvery"] = 1000;
    Warm["DipolesEvery"] = 100;
    Warm["StatusEvery"] = 100;
    Warm["WriteDensities"] = 0;
    Warm["FourierEvery"] = 0;
    Warm["SavePopulations"] = 1;
    Warm["SaveFockEnergies"] = 0;
    Warm["SaveFockEvery"] = 0;
    Warm["SerialNumber"] = 5;
    Warm["MaxIter"] = 0; // Perform an infinite number of propagations.

    NoField["ApplyImpulse"] = 0;
    NoField["Stabilize"] = 0;
    NoField["TCLOn"] = 0;
    NoField["StartFromSetman"] = 0;
    NoField["Stutter"] = 4500; // Pauses propagation every n iterations and does corrected fourier transform.
    NoField["UpdateFockMatrix"] = 1;
    NoField["SaveDipoles"] = 1;
    NoField["StatusEvery"] = 50;
    NoField["SaveEvery"] = 1000;
    NoField["WriteDensities"] = 0;
    NoField["FourierEvery"] = 0;
    NoField["SaveFockEnergies"] = 0;
    NoField["SaveFockEvery"] = 0;
    NoField["SerialNumber"] = 2;
    
    xField["UpdateFockMatrix"] = 1;
    xField["Stutter"] = 4500;
    xField["ApplyImpulse"] = 1; // Excited state absorption.
    xField["Stabilize"] = 0; // Excited state absorption.
    xField["TCLOn"] = 0; // Excited state absorption.
    xField["StatusEvery"] = 50;
    xField["SaveEvery"] = 1000;
    xField["StartFromSetman"] = 0; // The final density of Rlx will be fed in.
    xField["SavePopulations"] = 1;
    
    Gs["ApplyImpulse"] = 1;
    Gs["Stabilize"] = 2;
    Gs["TCLOn"] = 0;
    Gs["StartFromSetman"] = 0;
    Gs["Stutter"] = 4500;
    Gs["UpdateFockMatrix"] = 1;
    Gs["TCLOn"] = 0;
    Gs["SaveDipoles"] = 1;
    Gs["StatusEvery"] = 50;
    Gs["SaveEvery"] = 1000;
    Gs["WriteDensities"] = 0;
    Gs["FourierEvery"] = 0;
    Gs["SaveFockEnergies"] = 0;
    Gs["SaveFockEvery"] = 0;
    Gs["SerialNumber"] = 4;
  
    // perform multiple increasing delays for indefinite time, and sample TA spectrum every 500fs for the MaxIter in TDSCF.prm
    TDSCF* Gs_p = new TDSCF(pr,Gs,Scratch);
    double TSample = Gs_p->params["TSample"]/FsPerAu;
    Rlx["Stutter"] = (int)(TSample/Rlx["dt"]);
    double TWarm = Gs_p->params["TWarm"]/FsPerAu; // The warm job is really fast without any fock update.
    Warm["Stutter"] = (TWarm/Gs_p->params["dt"]); // Just long enough to decohere.
    cout << "----------------------------------------------------" << endl;
    cout << "Beginning pump probe propagations... " << endl;
    cout << "TWarm (au)" << TWarm << " NWarm:" << Warm["Stutter"] << endl;
    cout << "TSample (au)" << TSample << " NSample: " << Rlx["Stutter"] << endl;
    cout << "Excited Fraction " << Gs_p->params["ExcitedFraction"] << endl;
    cout << "----------------------------------------------------" << endl;
    
    int samples = 1;
    // The zero delay part doesn't require any Rlx, or warming.
    {
        Gs_p->Propagate();
        while (true)
        {
            Rlx["ExcitedFraction"] = exp(-samples/10.0);
            TDSCF* Rlx_p = new TDSCF(pr,Rlx,Scratch);
            Rlx_p->Rho = diagmat(Rlx_p->Rho);
            Rlx_p->RhoM12 = diagmat(Rlx_p->RhoM12);
            
            TDSCF* NonStationary_p = new TDSCF(pr,NoField,Scratch);
            xField["SerialNumber"] = 10+samples; // The zero delay propagation is named 3, 'cause.
            TDSCF* ToCorrect_p = new TDSCF(pr,xField,Scratch);
            NonStationary_p->Sync(Rlx_p);
            ToCorrect_p->Sync(Rlx_p);
            while (true)
            {
                if (!NonStationary_p->Propagate())
                    break;
                ToCorrect_p->Propagate();
                ToCorrect_p->TransientAbsorptionCorrection(NonStationary_p,Gs_p);
            }
            delete NonStationary_p;
            delete ToCorrect_p;
            delete Rlx_p;
            samples++;
        }
    }
    
    /*
    
    int samples = 1;
    TDSCF* Rlx_p = new TDSCF(pr,Rlx,Scratch);
    TDSCF* Warm_p = new TDSCF(pr,Warm,Scratch);
    // The zero delay part doesn't require any Rlx, or warming.
    {
        TDSCF* NonStationary_p = new TDSCF(pr,NoField,Scratch);
        xField["SerialNumber"] = 3; // The zero delay propagation is named 3, 'cause.
        TDSCF* ToCorrect_p = new TDSCF(pr,xField,Scratch);
        NonStationary_p->Sync(Rlx_p);
        ToCorrect_p->Sync(Rlx_p);
        while (true)
        {
            if (!Gs_p->Propagate())
                break;
            NonStationary_p->Propagate();
            ToCorrect_p->Propagate();
            ToCorrect_p->TransientAbsorptionCorrection(NonStationary_p,Gs_p);
        }
        delete NonStationary_p;
        delete ToCorrect_p;
    }
    // The finite delays require warming.
    while (true)
    {
        cout << "----------------------------------------------------" << endl;
        cout << "Beginning Finite Delay... tau = " << (TSample+TWarm)*samples*FsPerAu << "(fs)" << endl;
        cout << "----------------------------------------------------" << endl;
        if (!Rlx_p->Propagate())
            break;
        // Allow the density coming out of the relaxation job to settle for 10fs.
        Warm_p->Sync(Rlx_p);
        Warm_p->Propagate();
        Rlx_p->Sync(Warm_p);
        
        cout << "----------------------------------------------------" << endl;
        cout << "Relaxation Complete for " << samples*(TSample+TWarm)*FsPerAu << "(fs)" << endl;
        Rlx_p->Rho.print("Rho");
        cout << "----------------------------------------------------" << endl;
        // Grab the density from the warm job & pass it to these propagations.
        TDSCF* NonStationary_p = new TDSCF(pr,NoField,Scratch);
        xField["SerialNumber"] = (int)((TSample+TWarm)*samples*FsPerAu);
        TDSCF* ToCorrect_p = new TDSCF(pr,xField,Scratch);
        NonStationary_p->Sync(Warm_p);
        ToCorrect_p->Sync(Warm_p);
        while (true)
        {
            if (!NonStationary_p->Propagate())
                break;
            ToCorrect_p->Propagate();
            ToCorrect_p->TransientAbsorptionCorrection(NonStationary_p,Gs_p);
        }
        samples++;
        delete NonStationary_p;
        delete ToCorrect_p;
    }
     
    delete Warm_p;
    delete Gs_p;
    delete Rlx_p;
     */

    delete Scratch;
};

#endif
