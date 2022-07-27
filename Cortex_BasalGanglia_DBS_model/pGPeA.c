/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scoplib.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__GPeA
#define _nrn_initial _nrn_initial__GPeA
#define nrn_cur _nrn_cur__GPeA
#define _nrn_current _nrn_current__GPeA
#define nrn_jacob _nrn_jacob__GPeA
#define nrn_state _nrn_state__GPeA
#define _net_receive _net_receive__GPeA 
#define evaluate_fct evaluate_fct__GPeA 
#define states states__GPeA 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define gnabar _p[0]
#define gkdrbar _p[1]
#define gl _p[2]
#define el _p[3]
#define kca _p[4]
#define vol _p[5]
#define caGain _p[6]
#define gcatbar _p[7]
#define gkcabar _p[8]
#define ina _p[9]
#define ik _p[10]
#define ikD _p[11]
#define ikAHP _p[12]
#define ica _p[13]
#define icaT _p[14]
#define ilk _p[15]
#define h_inf _p[16]
#define tau_h _p[17]
#define m_inf _p[18]
#define tau_m _p[19]
#define ena _p[20]
#define n_inf _p[21]
#define tau_n _p[22]
#define ek _p[23]
#define p_inf _p[24]
#define q_inf _p[25]
#define tau_p _p[26]
#define tau_q _p[27]
#define eca _p[28]
#define r_inf _p[29]
#define m _p[30]
#define h _p[31]
#define n _p[32]
#define p _p[33]
#define q _p[34]
#define r _p[35]
#define Dm _p[36]
#define Dh _p[37]
#define Dn _p[38]
#define Dp _p[39]
#define Dq _p[40]
#define cai _p[41]
#define Dcai _p[42]
#define cao _p[43]
#define Dcao _p[44]
#define nai _p[45]
#define Dnai _p[46]
#define nao _p[47]
#define Dnao _p[48]
#define ki _p[49]
#define Dki _p[50]
#define ko _p[51]
#define Dko _p[52]
#define Dr _p[53]
#define v _p[54]
#define _g _p[55]
#define _ion_cai	*_ppvar[0]._pval
#define _ion_cao	*_ppvar[1]._pval
#define _ion_ica	*_ppvar[2]._pval
#define _ion_dicadv	*_ppvar[3]._pval
#define _style_ca	*((int*)_ppvar[4]._pvoid)
#define _ion_ki	*_ppvar[5]._pval
#define _ion_ko	*_ppvar[6]._pval
#define _ion_ik	*_ppvar[7]._pval
#define _ion_dikdv	*_ppvar[8]._pval
#define _ion_nai	*_ppvar[9]._pval
#define _ion_nao	*_ppvar[10]._pval
#define _ion_ina	*_ppvar[11]._pval
#define _ion_dinadv	*_ppvar[12]._pval
#define area	*_ppvar[13]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_evaluate_fct(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_GPeA", _hoc_setdata,
 "evaluate_fct_GPeA", _hoc_evaluate_fct,
 0, 0
};
 /* declare global and static user variables */
 static int _thread1data_inuse = 0;
static double _thread1data[1];
#define _gth 0
#define R R_GPeA
 double R = 8.31441;
#define T_GPeA _thread1data[0]
#define T _thread[_gth]._pval[0]
#define k_r k_r_GPeA
 double k_r = -8e-05;
#define k_q k_q_GPeA
 double k_q = 5.8;
#define k_p k_p_GPeA
 double k_p = -6.7;
#define k_n k_n_GPeA
 double k_n = -14;
#define k_h k_h_GPeA
 double k_h = 6.4;
#define k_m k_m_GPeA
 double k_m = -7;
#define power_r power_r_GPeA
 double power_r = 2;
#define sig_q2 sig_q2_GPeA
 double sig_q2 = 16;
#define sig_q1 sig_q1_GPeA
 double sig_q1 = -15;
#define sig_p2 sig_p2_GPeA
 double sig_p2 = 15;
#define sig_p1 sig_p1_GPeA
 double sig_p1 = -10;
#define sig_n2 sig_n2_GPeA
 double sig_n2 = 50;
#define sig_n1 sig_n1_GPeA
 double sig_n1 = -40;
#define sig_h2 sig_h2_GPeA
 double sig_h2 = 16;
#define sig_h1 sig_h1_GPeA
 double sig_h1 = -15;
#define sig_m sig_m_GPeA
 double sig_m = -0.7;
#define tau_r tau_r_GPeA
 double tau_r = 2;
#define theta_r theta_r_GPeA
 double theta_r = 0.00017;
#define tht_q2 tht_q2_GPeA
 double tht_q2 = -50;
#define tht_q1 tht_q1_GPeA
 double tht_q1 = -50;
#define tht_p2 tht_p2_GPeA
 double tht_p2 = -102;
#define tht_p1 tht_p1_GPeA
 double tht_p1 = -27;
#define tau_q1 tau_q1_GPeA
 double tau_q1 = 400;
#define tau_q0 tau_q0_GPeA
 double tau_q0 = 0;
#define tau_p1 tau_p1_GPeA
 double tau_p1 = 0.33;
#define tau_p0 tau_p0_GPeA
 double tau_p0 = 5;
#define theta_q theta_q_GPeA
 double theta_q = -85;
#define theta_p theta_p_GPeA
 double theta_p = -56;
#define tht_n2 tht_n2_GPeA
 double tht_n2 = -40;
#define tht_n1 tht_n1_GPeA
 double tht_n1 = -40;
#define tau_n1 tau_n1_GPeA
 double tau_n1 = 2.4;
#define tau_n0 tau_n0_GPeA
 double tau_n0 = 0;
#define theta_n theta_n_GPeA
 double theta_n = -42;
#define tht_h2 tht_h2_GPeA
 double tht_h2 = -50;
#define tht_h1 tht_h1_GPeA
 double tht_h1 = -50;
#define tht_m tht_m_GPeA
 double tht_m = -53;
#define tau_h1 tau_h1_GPeA
 double tau_h1 = 4.5;
#define tau_h0 tau_h0_GPeA
 double tau_h0 = 0;
#define tau_m1 tau_m1_GPeA
 double tau_m1 = 0.1;
#define tau_m0 tau_m0_GPeA
 double tau_m0 = 0.001;
#define theta_h theta_h_GPeA
 double theta_h = -45.5;
#define theta_m theta_m_GPeA
 double theta_m = -38;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "R_GPeA", "Gas",
 "T_GPeA", "Absolute",
 "theta_m_GPeA", "mV",
 "theta_h_GPeA", "mV",
 "k_m_GPeA", "mV",
 "k_h_GPeA", "mV",
 "tau_m0_GPeA", "ms",
 "tau_m1_GPeA", "ms",
 "tau_h0_GPeA", "ms",
 "tau_h1_GPeA", "ms",
 "tht_m_GPeA", "mV",
 "tht_h1_GPeA", "mV",
 "tht_h2_GPeA", "mV",
 "sig_m_GPeA", "mV",
 "sig_h1_GPeA", "mV",
 "sig_h2_GPeA", "mV",
 "theta_n_GPeA", "mV",
 "k_n_GPeA", "mV",
 "tau_n0_GPeA", "ms",
 "tau_n1_GPeA", "ms",
 "tht_n1_GPeA", "mV",
 "tht_n2_GPeA", "mV",
 "sig_n1_GPeA", "mV",
 "sig_n2_GPeA", "mV",
 "theta_p_GPeA", "mV",
 "theta_q_GPeA", "mV",
 "k_p_GPeA", "mV",
 "k_q_GPeA", "mV",
 "tau_p0_GPeA", "ms",
 "tau_p1_GPeA", "ms",
 "tau_q0_GPeA", "ms",
 "tau_q1_GPeA", "ms",
 "tht_p1_GPeA", "mV",
 "tht_p2_GPeA", "mV",
 "tht_q1_GPeA", "mV",
 "tht_q2_GPeA", "mV",
 "sig_p1_GPeA", "mV",
 "sig_p2_GPeA", "mV",
 "sig_q1_GPeA", "mV",
 "sig_q2_GPeA", "mV",
 "theta_r_GPeA", "mM",
 "k_r_GPeA", "mM",
 "tau_r_GPeA", "ms",
 "gnabar_GPeA", "S/cm2",
 "gkdrbar_GPeA", "S/cm2",
 "gl_GPeA", "S/cm2",
 "el_GPeA", "mV",
 "kca_GPeA", "1/ms",
 "vol_GPeA", "L",
 "gcatbar_GPeA", "S/cm2",
 "gkcabar_GPeA", "S/cm2",
 "ina_GPeA", "mA/cm2",
 "ik_GPeA", "mA/cm2",
 "ikD_GPeA", "mA/cm2",
 "ikAHP_GPeA", "mA/cm2",
 "ica_GPeA", "mA/cm2",
 "icaT_GPeA", "mA/cm2",
 "ilk_GPeA", "mA/cm2",
 "tau_h_GPeA", "ms",
 "tau_m_GPeA", "ms",
 "ena_GPeA", "mV",
 "tau_n_GPeA", "ms",
 "ek_GPeA", "mV",
 "tau_p_GPeA", "ms",
 "tau_q_GPeA", "ms",
 "eca_GPeA", "mV",
 0,0
};
 static double cao0 = 0;
 static double cai0 = 0;
 static double delta_t = 0.01;
 static double h0 = 0;
 static double ko0 = 0;
 static double ki0 = 0;
 static double m0 = 0;
 static double nao0 = 0;
 static double nai0 = 0;
 static double n0 = 0;
 static double p0 = 0;
 static double q0 = 0;
 static double r0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "R_GPeA", &R_GPeA,
 "T_GPeA", &T_GPeA,
 "theta_m_GPeA", &theta_m_GPeA,
 "theta_h_GPeA", &theta_h_GPeA,
 "k_m_GPeA", &k_m_GPeA,
 "k_h_GPeA", &k_h_GPeA,
 "tau_m0_GPeA", &tau_m0_GPeA,
 "tau_m1_GPeA", &tau_m1_GPeA,
 "tau_h0_GPeA", &tau_h0_GPeA,
 "tau_h1_GPeA", &tau_h1_GPeA,
 "tht_m_GPeA", &tht_m_GPeA,
 "tht_h1_GPeA", &tht_h1_GPeA,
 "tht_h2_GPeA", &tht_h2_GPeA,
 "sig_m_GPeA", &sig_m_GPeA,
 "sig_h1_GPeA", &sig_h1_GPeA,
 "sig_h2_GPeA", &sig_h2_GPeA,
 "theta_n_GPeA", &theta_n_GPeA,
 "k_n_GPeA", &k_n_GPeA,
 "tau_n0_GPeA", &tau_n0_GPeA,
 "tau_n1_GPeA", &tau_n1_GPeA,
 "tht_n1_GPeA", &tht_n1_GPeA,
 "tht_n2_GPeA", &tht_n2_GPeA,
 "sig_n1_GPeA", &sig_n1_GPeA,
 "sig_n2_GPeA", &sig_n2_GPeA,
 "theta_p_GPeA", &theta_p_GPeA,
 "theta_q_GPeA", &theta_q_GPeA,
 "k_p_GPeA", &k_p_GPeA,
 "k_q_GPeA", &k_q_GPeA,
 "tau_p0_GPeA", &tau_p0_GPeA,
 "tau_p1_GPeA", &tau_p1_GPeA,
 "tau_q0_GPeA", &tau_q0_GPeA,
 "tau_q1_GPeA", &tau_q1_GPeA,
 "tht_p1_GPeA", &tht_p1_GPeA,
 "tht_p2_GPeA", &tht_p2_GPeA,
 "tht_q1_GPeA", &tht_q1_GPeA,
 "tht_q2_GPeA", &tht_q2_GPeA,
 "sig_p1_GPeA", &sig_p1_GPeA,
 "sig_p2_GPeA", &sig_p2_GPeA,
 "sig_q1_GPeA", &sig_q1_GPeA,
 "sig_q2_GPeA", &sig_q2_GPeA,
 "theta_r_GPeA", &theta_r_GPeA,
 "k_r_GPeA", &k_r_GPeA,
 "tau_r_GPeA", &tau_r_GPeA,
 "power_r_GPeA", &power_r_GPeA,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[14]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"GPeA",
 "gnabar_GPeA",
 "gkdrbar_GPeA",
 "gl_GPeA",
 "el_GPeA",
 "kca_GPeA",
 "vol_GPeA",
 "caGain_GPeA",
 "gcatbar_GPeA",
 "gkcabar_GPeA",
 0,
 "ina_GPeA",
 "ik_GPeA",
 "ikD_GPeA",
 "ikAHP_GPeA",
 "ica_GPeA",
 "icaT_GPeA",
 "ilk_GPeA",
 "h_inf_GPeA",
 "tau_h_GPeA",
 "m_inf_GPeA",
 "tau_m_GPeA",
 "ena_GPeA",
 "n_inf_GPeA",
 "tau_n_GPeA",
 "ek_GPeA",
 "p_inf_GPeA",
 "q_inf_GPeA",
 "tau_p_GPeA",
 "tau_q_GPeA",
 "eca_GPeA",
 "r_inf_GPeA",
 0,
 "m_GPeA",
 "h_GPeA",
 "n_GPeA",
 "p_GPeA",
 "q_GPeA",
 "r_GPeA",
 0,
 0};
 extern Node* nrn_alloc_node_;
 static Symbol* _ca_sym;
 static Symbol* _k_sym;
 static Symbol* _na_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 56, _prop);
 	/*initialize range parameters*/
 	gnabar = 0.049;
 	gkdrbar = 0.057;
 	gl = 0.00035;
 	el = -60;
 	kca = 2;
 	vol = 3.355e-11;
 	caGain = 0.1;
 	gcatbar = 0.005;
 	gkcabar = 0.001;
 	_prop->param = _p;
 	_prop->param_size = 56;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 15, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 	_ppvar[13]._pval = &nrn_alloc_node_->_area; /* diam */
 prop_ion = need_memb(_ca_sym);
 nrn_check_conc_write(_prop, prop_ion, 1);
 nrn_promote(prop_ion, 3, 0);
 	_ppvar[0]._pval = &prop_ion->param[1]; /* cai */
 	_ppvar[1]._pval = &prop_ion->param[2]; /* cao */
 	_ppvar[2]._pval = &prop_ion->param[3]; /* ica */
 	_ppvar[3]._pval = &prop_ion->param[4]; /* _ion_dicadv */
 	_ppvar[4]._pvoid = (void*)(&(prop_ion->dparam[0]._i)); /* iontype for ca */
 prop_ion = need_memb(_k_sym);
 nrn_promote(prop_ion, 1, 0);
 	_ppvar[5]._pval = &prop_ion->param[1]; /* ki */
 	_ppvar[6]._pval = &prop_ion->param[2]; /* ko */
 	_ppvar[7]._pval = &prop_ion->param[3]; /* ik */
 	_ppvar[8]._pval = &prop_ion->param[4]; /* _ion_dikdv */
 prop_ion = need_memb(_na_sym);
 nrn_promote(prop_ion, 1, 0);
 	_ppvar[9]._pval = &prop_ion->param[1]; /* nai */
 	_ppvar[10]._pval = &prop_ion->param[2]; /* nao */
 	_ppvar[11]._pval = &prop_ion->param[3]; /* ina */
 	_ppvar[12]._pval = &prop_ion->param[4]; /* _ion_dinadv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _thread_mem_init(Datum*);
 static void _thread_cleanup(Datum*);
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _pGPeA_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("ca", -10000.);
 	ion_reg("k", -10000.);
 	ion_reg("na", -10000.);
 	_ca_sym = hoc_lookup("ca_ion");
 	_k_sym = hoc_lookup("k_ion");
 	_na_sym = hoc_lookup("na_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 2);
  _extcall_thread = (Datum*)ecalloc(1, sizeof(Datum));
  _thread_mem_init(_extcall_thread);
  _thread1data_inuse = 0;
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 1, _thread_mem_init);
     _nrn_thread_reg(_mechtype, 0, _thread_cleanup);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 56, 15);
  hoc_register_dparam_semantics(_mechtype, 0, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "#ca_ion");
  hoc_register_dparam_semantics(_mechtype, 5, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 6, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 7, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 8, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 9, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 10, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 11, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 12, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 14, "cvodeieq");
  hoc_register_dparam_semantics(_mechtype, 13, "area");
 	nrn_writes_conc(_mechtype, 0);
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 GPeA pGPeA.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 
#define FARADAY _nrnunit_FARADAY[_nrnunit_use_legacy_]
static double _nrnunit_FARADAY[2] = {0x1.78e555060882cp+16, 96485.3}; /* 96485.3321233100141 */
 
#define PI _nrnunit_PI[_nrnunit_use_legacy_]
static double _nrnunit_PI[2] = {0x1.921fb54442d18p+1, 3.14159}; /* 3.14159265358979312 */
static int _reset;
static char *modelname = "All ion channels used in GP models";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int evaluate_fct(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[7], _dlist1[7];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   evaluate_fct ( _threadargscomma_ v ) ;
   Dh = ( h_inf - h ) / tau_h ;
   Dm = ( m_inf - m ) / tau_m ;
   Dn = ( n_inf - n ) / tau_n ;
   Dp = ( p_inf - p ) / tau_p ;
   Dq = ( q_inf - q ) / tau_q ;
   Dcai = caGain * ( - ica * area * 1e-11 / ( 2.0 * FARADAY * vol ) - kca * cai ) ;
   Dr = ( r_inf - r ) / tau_r ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 evaluate_fct ( _threadargscomma_ v ) ;
 Dh = Dh  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau_h )) ;
 Dm = Dm  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau_m )) ;
 Dn = Dn  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau_n )) ;
 Dp = Dp  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau_p )) ;
 Dq = Dq  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau_q )) ;
 Dcai = Dcai  / (1. - dt*( ( caGain )*( ( ( - ( kca )*( 1.0 ) ) ) ) )) ;
 Dr = Dr  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau_r )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   evaluate_fct ( _threadargscomma_ v ) ;
    h = h + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau_h)))*(- ( ( ( h_inf ) ) / tau_h ) / ( ( ( ( - 1.0 ) ) ) / tau_h ) - h) ;
    m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau_m)))*(- ( ( ( m_inf ) ) / tau_m ) / ( ( ( ( - 1.0 ) ) ) / tau_m ) - m) ;
    n = n + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau_n)))*(- ( ( ( n_inf ) ) / tau_n ) / ( ( ( ( - 1.0 ) ) ) / tau_n ) - n) ;
    p = p + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau_p)))*(- ( ( ( p_inf ) ) / tau_p ) / ( ( ( ( - 1.0 ) ) ) / tau_p ) - p) ;
    q = q + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau_q)))*(- ( ( ( q_inf ) ) / tau_q ) / ( ( ( ( - 1.0 ) ) ) / tau_q ) - q) ;
    cai = cai + (1. - exp(dt*(( caGain )*( ( ( - ( kca )*( 1.0 ) ) ) ))))*(- ( ( caGain )*( ( ( ( ( - ica )*( area ) )*( 1e-11 ) ) / ( 2.0 * FARADAY * vol ) ) ) ) / ( ( caGain )*( ( ( - ( kca )*( 1.0 ) ) ) ) ) - cai) ;
    r = r + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau_r)))*(- ( ( ( r_inf ) ) / tau_r ) / ( ( ( ( - 1.0 ) ) ) / tau_r ) - r) ;
   }
  return 0;
}
 
static int  evaluate_fct ( _threadargsprotocomma_ double _lv ) {
   h_inf = 1.0 / ( 1.0 + exp ( ( _lv - theta_h ) / k_h ) ) ;
   m_inf = 1.0 / ( 1.0 + exp ( ( _lv - theta_m ) / k_m ) ) ;
   tau_h = tau_h0 + tau_h1 / ( exp ( - ( _lv - tht_h1 ) / sig_h1 ) + exp ( - ( _lv - tht_h2 ) / sig_h2 ) ) ;
   tau_m = tau_m0 + tau_m1 / ( 1.0 + exp ( - ( _lv - tht_m ) / sig_m ) ) ;
   n_inf = 1.0 / ( 1.0 + exp ( ( _lv - theta_n ) / k_n ) ) ;
   tau_n = tau_n0 + tau_n1 / ( exp ( - ( _lv - tht_n1 ) / sig_n1 ) + exp ( - ( _lv - tht_n2 ) / sig_n2 ) ) ;
   p_inf = 1.0 / ( 1.0 + exp ( ( _lv - theta_p ) / k_p ) ) ;
   q_inf = 1.0 / ( 1.0 + exp ( ( _lv - theta_q ) / k_q ) ) ;
   tau_p = tau_p0 + tau_p1 / ( exp ( - ( _lv - tht_p1 ) / sig_p1 ) + exp ( - ( _lv - tht_p2 ) / sig_p2 ) ) ;
   tau_q = tau_q0 + tau_q1 / ( exp ( - ( _lv - tht_q1 ) / sig_q1 ) + exp ( - ( _lv - tht_q2 ) / sig_q2 ) ) ;
   r_inf = 1.0 / ( 1.0 + exp ( ( cai - theta_r ) / k_r ) ) ;
    return 0; }
 
static void _hoc_evaluate_fct(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 evaluate_fct ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 7;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  cai = _ion_cai;
  cao = _ion_cao;
  cai = _ion_cai;
  ki = _ion_ki;
  ko = _ion_ko;
  nai = _ion_nai;
  nao = _ion_nao;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
   _ion_cai = cai;
   }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 7; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 	_pv[5] = &(_ion_cai);
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  cai = _ion_cai;
  cao = _ion_cao;
  cai = _ion_cai;
  ki = _ion_ki;
  ko = _ion_ko;
  nai = _ion_nai;
  nao = _ion_nao;
 _ode_matsol_instance1(_threadargs_);
 }}
 
static void _thread_mem_init(Datum* _thread) {
  if (_thread1data_inuse) {_thread[_gth]._pval = (double*)ecalloc(1, sizeof(double));
 }else{
 _thread[_gth]._pval = _thread1data; _thread1data_inuse = 1;
 }
 }
 
static void _thread_cleanup(Datum* _thread) {
  if (_thread[_gth]._pval == _thread1data) {
   _thread1data_inuse = 0;
  }else{
   free((void*)_thread[_gth]._pval);
  }
 }
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 0, 1);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 1, 2);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 2, 3);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 3, 4);
   nrn_update_ion_pointer(_k_sym, _ppvar, 5, 1);
   nrn_update_ion_pointer(_k_sym, _ppvar, 6, 2);
   nrn_update_ion_pointer(_k_sym, _ppvar, 7, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 8, 4);
   nrn_update_ion_pointer(_na_sym, _ppvar, 9, 1);
   nrn_update_ion_pointer(_na_sym, _ppvar, 10, 2);
   nrn_update_ion_pointer(_na_sym, _ppvar, 11, 3);
   nrn_update_ion_pointer(_na_sym, _ppvar, 12, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  h = h0;
  m = m0;
  n = n0;
  p = p0;
  q = q0;
  r = r0;
 {
   evaluate_fct ( _threadargscomma_ v ) ;
   m = m_inf ;
   h = h_inf ;
   n = n_inf ;
   p = p_inf ;
   q = q_inf ;
   r = r_inf ;
   }
 
}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
  cai = _ion_cai;
  cao = _ion_cao;
  cai = _ion_cai;
  ki = _ion_ki;
  ko = _ion_ko;
  nai = _ion_nai;
  nao = _ion_nao;
 initmodel(_p, _ppvar, _thread, _nt);
   _ion_cai = cai;
  nrn_wrote_conc(_ca_sym, (&(_ion_cai)) - 1, _style_ca);
  }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   T = 273.0 + celsius - 9.5 ;
   ena = - ( R * T ) / FARADAY * log ( nai / nao ) * 1000.0 ;
   ek = ( R * T ) / FARADAY * log ( ko / ki ) * 1000.0 ;
   eca = - ( R * T ) / FARADAY * log ( cai / cao ) * 1000.0 / 2.0 ;
   ina = gnabar * m * m * m * h * ( v - ena ) ;
   ikD = gkdrbar * pow( n , 4.0 ) * ( v - ek ) ;
   ikAHP = gkcabar * ( v - ek ) * pow( r , ( power_r ) ) ;
   ik = ikD + ikAHP ;
   ilk = gl * ( v - el ) ;
   ica = gcatbar * p * p * q * ( v - eca ) ;
   }
 _current += ilk;
 _current += ica;
 _current += ik;
 _current += ina;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
  cai = _ion_cai;
  cao = _ion_cao;
  cai = _ion_cai;
  ki = _ion_ki;
  ko = _ion_ko;
  nai = _ion_nai;
  nao = _ion_nao;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dina;
 double _dik;
 double _dica;
  _dica = ica;
  _dik = ik;
  _dina = ina;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dicadv += (_dica - ica)/.001 ;
  _ion_dikdv += (_dik - ik)/.001 ;
  _ion_dinadv += (_dina - ina)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ica += ica ;
  _ion_cai = cai;
  _ion_ik += ik ;
  _ion_ina += ina ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
  cai = _ion_cai;
  cao = _ion_cao;
  cai = _ion_cai;
  ki = _ion_ki;
  ko = _ion_ko;
  nai = _ion_nai;
  nao = _ion_nao;
 {   states(_p, _ppvar, _thread, _nt);
  }   _ion_cai = cai;
  }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = &(h) - _p;  _dlist1[0] = &(Dh) - _p;
 _slist1[1] = &(m) - _p;  _dlist1[1] = &(Dm) - _p;
 _slist1[2] = &(n) - _p;  _dlist1[2] = &(Dn) - _p;
 _slist1[3] = &(p) - _p;  _dlist1[3] = &(Dp) - _p;
 _slist1[4] = &(q) - _p;  _dlist1[4] = &(Dq) - _p;
 _slist1[5] = &(cai) - _p;  _dlist1[5] = &(Dcai) - _p;
 _slist1[6] = &(r) - _p;  _dlist1[6] = &(Dr) - _p;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "pGPeA.mod";
static const char* nmodl_file_text = 
  "TITLE  All ion channels used in GP models\n"
  "\n"
  ":\n"
  ": Na+, K, Ca_T, Leakage, AHP and Ca diffusion\n"
  ": \n"
  "\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX GPeA\n"
  "	NONSPECIFIC_CURRENT ilk\n"
  "	USEION ca READ cai, cao WRITE ica, cai\n"
  "	USEION k READ ki, ko WRITE ik\n"
  "	USEION na READ nai, nao WRITE ina\n"
  "	RANGE ina, ik, ica\n"
  "	RANGE gnabar, ena, m_inf, h_inf, tau_h, tau_m        : fast sodium\n"
  "	RANGE gkdrbar, ek, n_inf, tau_n, ikD                 : delayed K rectifier\n"
  "	RANGE gl, el, ilk                                    : leak\n"
  "	RANGE gcatbar, eca, p_inf, tau_p, q_inf, tau_q, icaT : T-type ca current\n"
  "	RANGE gkcabar, ek, r_inf, ikAHP                      : ca dependent AHP K current\n"
  "      	RANGE kca, vol, caGain                               : ca dynamics\n"
  "}\n"
  "\n"
  "\n"
  "UNITS {\n"
  "	(mA) = (milliamp)\n"
  "	(mV) = (millivolt)\n"
  "	(S)  = (siemens)\n"
  "	(molar) = (1/liter)\n"
  "	(mM)	= (millimolar)\n"
  "	FARADAY = (faraday) (coulomb)  :units are really coulombs/mole\n"
  "	PI	= (pi) (1)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	R = 8.31441 (Gas constant)\n"
  "	T 		(Absolute temp)\n"
  "	celsius		(degC)\n"
  "\n"
  ":Fast Na channel\n"
  "	gnabar   = 49e-3 (S/cm2) \n"
  "	theta_m = -38 (mV)\n"
  "	theta_h = -45.5 (mV) \n"
  "	k_m = -7 (mV)    \n"
  "	k_h = 6.4 (mV)   \n"
  "	tau_m0 = 0.001 (ms)\n"
  "	tau_m1 = 0.1 (ms)\n"
  "	tau_h0 = 0 (ms)\n"
  "	tau_h1 = 4.5 (ms) \n"
  "	tht_m = -53 (mV)\n"
  "	tht_h1 = -50 (mV)\n"
  "	tht_h2 = -50 (mV)\n"
  "	sig_m = -0.7 (mV)\n"
  "	sig_h1 = -15 (mV)\n"
  "	sig_h2 = 16 (mV)\n"
  "\n"
  ": delayed K rectifier \n"
  "	gkdrbar  = 57e-3	(S/cm2)  \n"
  "	theta_n = -42 (mV)\n"
  "	k_n = -14 (mV)     \n"
  "	tau_n0 = 0 (ms)\n"
  "	tau_n1 = 2.4 (ms) \n"
  "	tht_n1 = -40 (mV)\n"
  "	tht_n2 = -40 (mV)\n"
  "	sig_n1 = -40 (mV)\n"
  "	sig_n2 = 50 (mV)\n"
  "\n"
  ":Leakage current\n"
  "	gl	= 0.35e-3	(S/cm2)\n"
  "	el	= -60	(mV)\n"
  "\n"
  ":Ca dynamics\n"
  "	kca   = 2        (1/ms)\n"
  "      	area\n"
  "      	vol = 3.355e-11  (L) :~20um radius sphere\n"
  "      	caGain = .1\n"
  "\n"
  ":T-type ca current\n"
  "	gcatbar   = 5e-3 (S/cm2)  \n"
  "	theta_p = -56 (mV)\n"
  "	theta_q = -85 (mV) \n"
  "	k_p = -6.7 (mV)    \n"
  "	k_q = 5.8 (mV)  \n"
  "	tau_p0 = 5 (ms)\n"
  "	tau_p1 = 0.33 (ms)\n"
  "	tau_q0 = 0 (ms)\n"
  "	tau_q1 = 400 (ms) \n"
  "	tht_p1 = -27 (mV)\n"
  "	tht_p2 = -102 (mV)\n"
  "	tht_q1 = -50 (mV)\n"
  "	tht_q2 = -50 (mV)\n"
  "	sig_p1 = -10 (mV)\n"
  "	sig_p2 = 15 (mV)\n"
  "	sig_q1 = -15 (mV)\n"
  "	sig_q2 = 16 (mV)\n"
  "\n"
  ":AHP current (Ca dependent K current)\n"
  "	gkcabar   = 1e-3 (S/cm2) \n"
  "	theta_r = 0.17e-3 (mM)\n"
  "	k_r = -0.08e-3 (mM)\n"
  "	tau_r = 2 (ms)\n"
  "	power_r = 2\n"
  "\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	v	(mV)\n"
  "	ina	(mA/cm2)\n"
  "	ik	(mA/cm2) \n"
  "	ikD	(mA/cm2)   \n"
  "	ikAHP	(mA/cm2)  \n"
  "	ica	(mA/cm2) \n"
  "	icaT	(mA/cm2) \n"
  "	ilk	(mA/cm2)\n"
  "\n"
  ":Fast Na\n"
  "	h_inf\n"
  "	tau_h	(ms)\n"
  "	m_inf\n"
  "	tau_m	(ms)\n"
  "	ena           (mV)   := 60\n"
  "\n"
  ":K rectifier\n"
  "	n_inf\n"
  "	tau_n	(ms)\n"
  "	ek         (mV) := -90\n"
  "\n"
  ":T-type ca current\n"
  "	p_inf\n"
  "	q_inf\n"
  "	tau_p	(ms)\n"
  "	tau_q	(ms)\n"
  "	eca           (mV)   :calc from Nernst\n"
  "\n"
  ":AHP (Ca dependent K current)\n"
  "	r_inf\n"
  "}\n"
  "\n"
  "STATE {\n"
  "	m h n \n"
  "	p q\n"
  "	cai (mM) <1e-10>\n"
  "	cao (mM) <1e-10>\n"
  "	nai (mM) <1e-10>\n"
  "	nao (mM) <1e-10>\n"
  "	ki (mM) <1e-10>\n"
  "	ko (mM) <1e-10>\n"
  "      r\n"
  "}\n"
  "\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE states METHOD cnexp\n"
  "\n"
  "	T = 273 + celsius - 9.5\n"
  "	ena = -(R*T)/FARADAY*log(nai/nao)*1000\n"
  "	ek = (R*T)/FARADAY*log(ko/ki)*1000\n"
  "	eca = -(R*T)/FARADAY*log(cai/cao)*1000/2\n"
  "\n"
  "	ina = gnabar * m*m*m*h * (v - ena)\n"
  "	ikD = gkdrbar * n^4 * (v - ek)\n"
  "	ikAHP = gkcabar * (v - ek)*r^(power_r)\n"
  "	ik=ikD+ikAHP\n"
  "	ilk = gl * (v - el)\n"
  "	ica = gcatbar * p*p*q * (v - eca)\n"
  "}\n"
  "\n"
  "DERIVATIVE states {   \n"
  "	evaluate_fct(v)\n"
  "	h' = (h_inf - h)/tau_h\n"
  "	m' = (m_inf - m)/tau_m\n"
  "	n' = (n_inf - n)/tau_n\n"
  "	p' = (p_inf - p)/tau_p\n"
  "	q' = (q_inf - q)/tau_q\n"
  "\n"
  "      :(Ica mA/cm2)*(area um2)*(1e-8 cm2/um2)*(1e-3 A/mA)*(1/(2*F) mol/C)*(1e-3 sec/msec)*(1e3 mMol/mol)(1/volume 1/L)=(mM/msec)\n"
  "	cai' = caGain*(-ica*area*1e-11/(2*FARADAY*vol) - kca*cai)\n"
  "\n"
  "	r' = (r_inf - r)/tau_r\n"
  "}\n"
  "\n"
  "UNITSOFF\n"
  "\n"
  "INITIAL {\n"
  "	evaluate_fct(v)\n"
  "	m = m_inf \n"
  "	h = h_inf \n"
  "	n = n_inf   \n"
  "	p = p_inf \n"
  "	q = q_inf   \n"
  "	r = r_inf \n"
  "}\n"
  "\n"
  "PROCEDURE evaluate_fct(v(mV)) { \n"
  "\n"
  "	h_inf = 1/(1+exp((v-theta_h)/k_h))\n"
  "	m_inf = 1/(1+exp((v-theta_m)/k_m))\n"
  "	tau_h = tau_h0 + tau_h1/(exp(-(v-tht_h1)/sig_h1) + exp(-(v-tht_h2)/sig_h2)) \n"
  "	tau_m = tau_m0 + tau_m1/(1+exp(-(v-tht_m)/sig_m)) \n"
  "\n"
  "	n_inf = 1/(1+exp((v-theta_n)/k_n))\n"
  "	tau_n = tau_n0 + tau_n1/(exp(-(v-tht_n1)/sig_n1) + exp(-(v-tht_n2)/sig_n2))\n"
  "\n"
  "	p_inf = 1/(1+exp((v-theta_p)/k_p))\n"
  "	q_inf = 1/(1+exp((v-theta_q)/k_q))\n"
  "	tau_p = tau_p0 + tau_p1/(exp(-(v-tht_p1)/sig_p1) + exp(-(v-tht_p2)/sig_p2)) \n"
  "	tau_q = tau_q0 + tau_q1/(exp(-(v-tht_q1)/sig_q1) + exp(-(v-tht_q2)/sig_q2)) \n"
  "\n"
  "	r_inf = 1/(1+exp((cai-theta_r)/k_r))\n"
  "}\n"
  "\n"
  "UNITSON\n"
  ;
#endif
