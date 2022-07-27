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
 
#define nrn_init _nrn_init__stn
#define _nrn_initial _nrn_initial__stn
#define nrn_cur _nrn_cur__stn
#define _nrn_current _nrn_current__stn
#define nrn_jacob _nrn_jacob__stn
#define nrn_state _nrn_state__stn
#define _net_receive _net_receive__stn 
#define evaluate_fct2 evaluate_fct2__stn 
#define evaluate_fct evaluate_fct__stn 
#define states states__stn 
 
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
#define gcalbar _p[8]
#define tau_d2 _p[9]
#define gkabar _p[10]
#define gkcabar _p[11]
#define ina _p[12]
#define ik _p[13]
#define ikD _p[14]
#define ikA _p[15]
#define ikAHP _p[16]
#define ica _p[17]
#define icaT _p[18]
#define icaL _p[19]
#define ilk _p[20]
#define h_inf _p[21]
#define tau_h _p[22]
#define m_inf _p[23]
#define tau_m _p[24]
#define ena _p[25]
#define n_inf _p[26]
#define tau_n _p[27]
#define ek _p[28]
#define p_inf _p[29]
#define q_inf _p[30]
#define tau_p _p[31]
#define tau_q _p[32]
#define eca _p[33]
#define c_inf _p[34]
#define tau_c _p[35]
#define d1_inf _p[36]
#define tau_d1 _p[37]
#define d2_inf _p[38]
#define a_inf _p[39]
#define tau_a _p[40]
#define b_inf _p[41]
#define tau_b _p[42]
#define r_inf _p[43]
#define m _p[44]
#define h _p[45]
#define n _p[46]
#define p _p[47]
#define q _p[48]
#define c _p[49]
#define d1 _p[50]
#define d2 _p[51]
#define a _p[52]
#define b _p[53]
#define r _p[54]
#define Dm _p[55]
#define Dh _p[56]
#define Dn _p[57]
#define Dp _p[58]
#define Dq _p[59]
#define Dc _p[60]
#define Dd1 _p[61]
#define Dd2 _p[62]
#define cai _p[63]
#define Dcai _p[64]
#define cao _p[65]
#define Dcao _p[66]
#define nai _p[67]
#define Dnai _p[68]
#define nao _p[69]
#define Dnao _p[70]
#define ki _p[71]
#define Dki _p[72]
#define ko _p[73]
#define Dko _p[74]
#define Da _p[75]
#define Db _p[76]
#define Dr _p[77]
#define v _p[78]
#define _g _p[79]
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
 static void _hoc_evaluate_fct2(void);
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
 "setdata_stn", _hoc_setdata,
 "evaluate_fct2_stn", _hoc_evaluate_fct2,
 "evaluate_fct_stn", _hoc_evaluate_fct,
 0, 0
};
 /* declare global and static user variables */
 static int _thread1data_inuse = 0;
static double _thread1data[1];
#define _gth 0
#define R R_stn
 double R = 8.31441;
#define T_stn _thread1data[0]
#define T _thread[_gth]._pval[0]
#define k_r k_r_stn
 double k_r = -8e-05;
#define k_b k_b_stn
 double k_b = 7.5;
#define k_a k_a_stn
 double k_a = -14.7;
#define k_d2 k_d2_stn
 double k_d2 = 2e-05;
#define k_d1 k_d1_stn
 double k_d1 = 7.5;
#define k_c k_c_stn
 double k_c = -5;
#define k_q k_q_stn
 double k_q = 5.8;
#define k_p k_p_stn
 double k_p = -6.7;
#define k_n k_n_stn
 double k_n = -14;
#define k_h k_h_stn
 double k_h = 6.4;
#define k_m k_m_stn
 double k_m = -8;
#define power_r power_r_stn
 double power_r = 2;
#define sig_b2 sig_b2_stn
 double sig_b2 = 10;
#define sig_b1 sig_b1_stn
 double sig_b1 = -30;
#define sig_a sig_a_stn
 double sig_a = -0.5;
#define sig_d12 sig_d12_stn
 double sig_d12 = 20;
#define sig_d11 sig_d11_stn
 double sig_d11 = -15;
#define sig_c2 sig_c2_stn
 double sig_c2 = 15;
#define sig_c1 sig_c1_stn
 double sig_c1 = -20;
#define sig_q2 sig_q2_stn
 double sig_q2 = 16;
#define sig_q1 sig_q1_stn
 double sig_q1 = -15;
#define sig_p2 sig_p2_stn
 double sig_p2 = 15;
#define sig_p1 sig_p1_stn
 double sig_p1 = -10;
#define sig_n2 sig_n2_stn
 double sig_n2 = 50;
#define sig_n1 sig_n1_stn
 double sig_n1 = -40;
#define sig_h2 sig_h2_stn
 double sig_h2 = 16;
#define sig_h1 sig_h1_stn
 double sig_h1 = -15;
#define sig_m sig_m_stn
 double sig_m = -0.7;
#define tau_r tau_r_stn
 double tau_r = 2;
#define theta_r theta_r_stn
 double theta_r = 0.00017;
#define tht_b2 tht_b2_stn
 double tht_b2 = -40;
#define tht_b1 tht_b1_stn
 double tht_b1 = -60;
#define tht_a tht_a_stn
 double tht_a = -40;
#define tau_b1 tau_b1_stn
 double tau_b1 = 200;
#define tau_b0 tau_b0_stn
 double tau_b0 = 0;
#define tau_a1 tau_a1_stn
 double tau_a1 = 1;
#define tau_a0 tau_a0_stn
 double tau_a0 = 1;
#define theta_b theta_b_stn
 double theta_b = -90;
#define theta_a theta_a_stn
 double theta_a = -45;
#define tht_d12 tht_d12_stn
 double tht_d12 = -20;
#define tht_d11 tht_d11_stn
 double tht_d11 = -40;
#define tht_c2 tht_c2_stn
 double tht_c2 = -50;
#define tht_c1 tht_c1_stn
 double tht_c1 = -27;
#define tau_d11 tau_d11_stn
 double tau_d11 = 500;
#define tau_d10 tau_d10_stn
 double tau_d10 = 400;
#define tau_c1 tau_c1_stn
 double tau_c1 = 10;
#define tau_c0 tau_c0_stn
 double tau_c0 = 45;
#define theta_d2 theta_d2_stn
 double theta_d2 = 0.0001;
#define theta_d1 theta_d1_stn
 double theta_d1 = -60;
#define theta_c theta_c_stn
 double theta_c = -30.6;
#define tht_q2 tht_q2_stn
 double tht_q2 = -50;
#define tht_q1 tht_q1_stn
 double tht_q1 = -50;
#define tht_p2 tht_p2_stn
 double tht_p2 = -102;
#define tht_p1 tht_p1_stn
 double tht_p1 = -27;
#define tau_q1 tau_q1_stn
 double tau_q1 = 400;
#define tau_q0 tau_q0_stn
 double tau_q0 = 0;
#define tau_p1 tau_p1_stn
 double tau_p1 = 0.33;
#define tau_p0 tau_p0_stn
 double tau_p0 = 5;
#define theta_q theta_q_stn
 double theta_q = -85;
#define theta_p theta_p_stn
 double theta_p = -56;
#define tht_n2 tht_n2_stn
 double tht_n2 = -40;
#define tht_n1 tht_n1_stn
 double tht_n1 = -40;
#define tau_n1 tau_n1_stn
 double tau_n1 = 11;
#define tau_n0 tau_n0_stn
 double tau_n0 = 0;
#define theta_n theta_n_stn
 double theta_n = -41;
#define tht_h2 tht_h2_stn
 double tht_h2 = -50;
#define tht_h1 tht_h1_stn
 double tht_h1 = -50;
#define tht_m tht_m_stn
 double tht_m = -53;
#define tau_h1 tau_h1_stn
 double tau_h1 = 24.5;
#define tau_h0 tau_h0_stn
 double tau_h0 = 0;
#define tau_m1 tau_m1_stn
 double tau_m1 = 3;
#define tau_m0 tau_m0_stn
 double tau_m0 = 0.2;
#define theta_h theta_h_stn
 double theta_h = -45.5;
#define theta_m theta_m_stn
 double theta_m = -40;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "R_stn", "Gas",
 "T_stn", "Absolute",
 "theta_m_stn", "mV",
 "theta_h_stn", "mV",
 "k_m_stn", "mV",
 "k_h_stn", "mV",
 "tau_m0_stn", "ms",
 "tau_m1_stn", "ms",
 "tau_h0_stn", "ms",
 "tau_h1_stn", "ms",
 "tht_m_stn", "mV",
 "tht_h1_stn", "mV",
 "tht_h2_stn", "mV",
 "sig_m_stn", "mV",
 "sig_h1_stn", "mV",
 "sig_h2_stn", "mV",
 "theta_n_stn", "mV",
 "k_n_stn", "mV",
 "tau_n0_stn", "ms",
 "tau_n1_stn", "ms",
 "tht_n1_stn", "mV",
 "tht_n2_stn", "mV",
 "sig_n1_stn", "mV",
 "sig_n2_stn", "mV",
 "theta_p_stn", "mV",
 "theta_q_stn", "mV",
 "k_p_stn", "mV",
 "k_q_stn", "mV",
 "tau_p0_stn", "ms",
 "tau_p1_stn", "ms",
 "tau_q0_stn", "ms",
 "tau_q1_stn", "ms",
 "tht_p1_stn", "mV",
 "tht_p2_stn", "mV",
 "tht_q1_stn", "mV",
 "tht_q2_stn", "mV",
 "sig_p1_stn", "mV",
 "sig_p2_stn", "mV",
 "sig_q1_stn", "mV",
 "sig_q2_stn", "mV",
 "theta_c_stn", "mV",
 "theta_d1_stn", "mV",
 "theta_d2_stn", "mM",
 "k_c_stn", "mV",
 "k_d1_stn", "mV",
 "k_d2_stn", "mM",
 "tau_c0_stn", "ms",
 "tau_c1_stn", "ms",
 "tau_d10_stn", "ms",
 "tau_d11_stn", "ms",
 "tht_c1_stn", "mV",
 "tht_c2_stn", "mV",
 "tht_d11_stn", "mV",
 "tht_d12_stn", "mV",
 "sig_c1_stn", "mV",
 "sig_c2_stn", "mV",
 "sig_d11_stn", "mV",
 "sig_d12_stn", "mV",
 "theta_a_stn", "mV",
 "theta_b_stn", "mV",
 "k_a_stn", "mV",
 "k_b_stn", "mV",
 "tau_a0_stn", "ms",
 "tau_a1_stn", "ms",
 "tau_b0_stn", "ms",
 "tau_b1_stn", "ms",
 "tht_a_stn", "mV",
 "tht_b1_stn", "mV",
 "tht_b2_stn", "mV",
 "sig_a_stn", "mV",
 "sig_b1_stn", "mV",
 "sig_b2_stn", "mV",
 "theta_r_stn", "mM",
 "k_r_stn", "mM",
 "tau_r_stn", "ms",
 "gnabar_stn", "S/cm2",
 "gkdrbar_stn", "S/cm2",
 "gl_stn", "S/cm2",
 "el_stn", "mV",
 "kca_stn", "1/ms",
 "vol_stn", "L",
 "gcatbar_stn", "S/cm2",
 "gcalbar_stn", "S/cm2",
 "tau_d2_stn", "ms",
 "gkabar_stn", "S/cm2",
 "gkcabar_stn", "S/cm2",
 "ina_stn", "mA/cm2",
 "ik_stn", "mA/cm2",
 "ikD_stn", "mA/cm2",
 "ikA_stn", "mA/cm2",
 "ikAHP_stn", "mA/cm2",
 "ica_stn", "mA/cm2",
 "icaT_stn", "mA/cm2",
 "icaL_stn", "mA/cm2",
 "ilk_stn", "mA/cm2",
 "tau_h_stn", "ms",
 "tau_m_stn", "ms",
 "ena_stn", "mV",
 "tau_n_stn", "ms",
 "ek_stn", "mV",
 "tau_p_stn", "ms",
 "tau_q_stn", "ms",
 "eca_stn", "mV",
 "tau_c_stn", "ms",
 "tau_d1_stn", "ms",
 "tau_a_stn", "ms",
 "tau_b_stn", "ms",
 0,0
};
 static double a0 = 0;
 static double b0 = 0;
 static double cao0 = 0;
 static double cai0 = 0;
 static double c0 = 0;
 static double delta_t = 0.01;
 static double d20 = 0;
 static double d10 = 0;
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
 "R_stn", &R_stn,
 "T_stn", &T_stn,
 "theta_m_stn", &theta_m_stn,
 "theta_h_stn", &theta_h_stn,
 "k_m_stn", &k_m_stn,
 "k_h_stn", &k_h_stn,
 "tau_m0_stn", &tau_m0_stn,
 "tau_m1_stn", &tau_m1_stn,
 "tau_h0_stn", &tau_h0_stn,
 "tau_h1_stn", &tau_h1_stn,
 "tht_m_stn", &tht_m_stn,
 "tht_h1_stn", &tht_h1_stn,
 "tht_h2_stn", &tht_h2_stn,
 "sig_m_stn", &sig_m_stn,
 "sig_h1_stn", &sig_h1_stn,
 "sig_h2_stn", &sig_h2_stn,
 "theta_n_stn", &theta_n_stn,
 "k_n_stn", &k_n_stn,
 "tau_n0_stn", &tau_n0_stn,
 "tau_n1_stn", &tau_n1_stn,
 "tht_n1_stn", &tht_n1_stn,
 "tht_n2_stn", &tht_n2_stn,
 "sig_n1_stn", &sig_n1_stn,
 "sig_n2_stn", &sig_n2_stn,
 "theta_p_stn", &theta_p_stn,
 "theta_q_stn", &theta_q_stn,
 "k_p_stn", &k_p_stn,
 "k_q_stn", &k_q_stn,
 "tau_p0_stn", &tau_p0_stn,
 "tau_p1_stn", &tau_p1_stn,
 "tau_q0_stn", &tau_q0_stn,
 "tau_q1_stn", &tau_q1_stn,
 "tht_p1_stn", &tht_p1_stn,
 "tht_p2_stn", &tht_p2_stn,
 "tht_q1_stn", &tht_q1_stn,
 "tht_q2_stn", &tht_q2_stn,
 "sig_p1_stn", &sig_p1_stn,
 "sig_p2_stn", &sig_p2_stn,
 "sig_q1_stn", &sig_q1_stn,
 "sig_q2_stn", &sig_q2_stn,
 "theta_c_stn", &theta_c_stn,
 "theta_d1_stn", &theta_d1_stn,
 "theta_d2_stn", &theta_d2_stn,
 "k_c_stn", &k_c_stn,
 "k_d1_stn", &k_d1_stn,
 "k_d2_stn", &k_d2_stn,
 "tau_c0_stn", &tau_c0_stn,
 "tau_c1_stn", &tau_c1_stn,
 "tau_d10_stn", &tau_d10_stn,
 "tau_d11_stn", &tau_d11_stn,
 "tht_c1_stn", &tht_c1_stn,
 "tht_c2_stn", &tht_c2_stn,
 "tht_d11_stn", &tht_d11_stn,
 "tht_d12_stn", &tht_d12_stn,
 "sig_c1_stn", &sig_c1_stn,
 "sig_c2_stn", &sig_c2_stn,
 "sig_d11_stn", &sig_d11_stn,
 "sig_d12_stn", &sig_d12_stn,
 "theta_a_stn", &theta_a_stn,
 "theta_b_stn", &theta_b_stn,
 "k_a_stn", &k_a_stn,
 "k_b_stn", &k_b_stn,
 "tau_a0_stn", &tau_a0_stn,
 "tau_a1_stn", &tau_a1_stn,
 "tau_b0_stn", &tau_b0_stn,
 "tau_b1_stn", &tau_b1_stn,
 "tht_a_stn", &tht_a_stn,
 "tht_b1_stn", &tht_b1_stn,
 "tht_b2_stn", &tht_b2_stn,
 "sig_a_stn", &sig_a_stn,
 "sig_b1_stn", &sig_b1_stn,
 "sig_b2_stn", &sig_b2_stn,
 "theta_r_stn", &theta_r_stn,
 "k_r_stn", &k_r_stn,
 "tau_r_stn", &tau_r_stn,
 "power_r_stn", &power_r_stn,
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
"stn",
 "gnabar_stn",
 "gkdrbar_stn",
 "gl_stn",
 "el_stn",
 "kca_stn",
 "vol_stn",
 "caGain_stn",
 "gcatbar_stn",
 "gcalbar_stn",
 "tau_d2_stn",
 "gkabar_stn",
 "gkcabar_stn",
 0,
 "ina_stn",
 "ik_stn",
 "ikD_stn",
 "ikA_stn",
 "ikAHP_stn",
 "ica_stn",
 "icaT_stn",
 "icaL_stn",
 "ilk_stn",
 "h_inf_stn",
 "tau_h_stn",
 "m_inf_stn",
 "tau_m_stn",
 "ena_stn",
 "n_inf_stn",
 "tau_n_stn",
 "ek_stn",
 "p_inf_stn",
 "q_inf_stn",
 "tau_p_stn",
 "tau_q_stn",
 "eca_stn",
 "c_inf_stn",
 "tau_c_stn",
 "d1_inf_stn",
 "tau_d1_stn",
 "d2_inf_stn",
 "a_inf_stn",
 "tau_a_stn",
 "b_inf_stn",
 "tau_b_stn",
 "r_inf_stn",
 0,
 "m_stn",
 "h_stn",
 "n_stn",
 "p_stn",
 "q_stn",
 "c_stn",
 "d1_stn",
 "d2_stn",
 "a_stn",
 "b_stn",
 "r_stn",
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
 	_p = nrn_prop_data_alloc(_mechtype, 80, _prop);
 	/*initialize range parameters*/
 	gnabar = 0.049;
 	gkdrbar = 0.057;
 	gl = 0.00035;
 	el = -60;
 	kca = 2;
 	vol = 3.355e-11;
 	caGain = 0.1;
 	gcatbar = 0.005;
 	gcalbar = 0.015;
 	tau_d2 = 130;
 	gkabar = 0.005;
 	gkcabar = 0.001;
 	_prop->param = _p;
 	_prop->param_size = 80;
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

 void _pSTN_reg() {
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
  hoc_register_prop_size(_mechtype, 80, 15);
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
 	ivoc_help("help ?1 stn pSTN.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 
#define FARADAY _nrnunit_FARADAY[_nrnunit_use_legacy_]
static double _nrnunit_FARADAY[2] = {0x1.78e555060882cp+16, 96485.3}; /* 96485.3321233100141 */
static int _reset;
static char *modelname = "STN ion channels for single compartment model";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int evaluate_fct2(_threadargsprotocomma_ double);
static int evaluate_fct(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[12], _dlist1[12];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   evaluate_fct ( _threadargscomma_ v ) ;
   Dh = ( h_inf - h ) / tau_h ;
   Dm = ( m_inf - m ) / tau_m ;
   Dn = ( n_inf - n ) / tau_n ;
   Dp = ( p_inf - p ) / tau_p ;
   Dq = ( q_inf - q ) / tau_q ;
   evaluate_fct2 ( _threadargscomma_ cai ) ;
   Dc = ( c_inf - c ) / tau_c ;
   Dd1 = ( d1_inf - d1 ) / tau_d1 ;
   Dd2 = ( d2_inf - d2 ) / tau_d2 ;
   Dcai = caGain * ( - ica * area * 1e-11 / ( 2.0 * FARADAY * vol ) - kca * cai ) ;
   Da = ( a_inf - a ) / tau_a ;
   Db = ( b_inf - b ) / tau_b ;
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
 evaluate_fct2 ( _threadargscomma_ cai ) ;
 Dc = Dc  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau_c )) ;
 Dd1 = Dd1  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau_d1 )) ;
 Dd2 = Dd2  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau_d2 )) ;
 Dcai = Dcai  / (1. - dt*( ( caGain )*( ( ( - ( kca )*( 1.0 ) ) ) ) )) ;
 Da = Da  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau_a )) ;
 Db = Db  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau_b )) ;
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
   evaluate_fct2 ( _threadargscomma_ cai ) ;
    c = c + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau_c)))*(- ( ( ( c_inf ) ) / tau_c ) / ( ( ( ( - 1.0 ) ) ) / tau_c ) - c) ;
    d1 = d1 + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau_d1)))*(- ( ( ( d1_inf ) ) / tau_d1 ) / ( ( ( ( - 1.0 ) ) ) / tau_d1 ) - d1) ;
    d2 = d2 + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau_d2)))*(- ( ( ( d2_inf ) ) / tau_d2 ) / ( ( ( ( - 1.0 ) ) ) / tau_d2 ) - d2) ;
    cai = cai + (1. - exp(dt*(( caGain )*( ( ( - ( kca )*( 1.0 ) ) ) ))))*(- ( ( caGain )*( ( ( ( ( - ica )*( area ) )*( 1e-11 ) ) / ( 2.0 * FARADAY * vol ) ) ) ) / ( ( caGain )*( ( ( - ( kca )*( 1.0 ) ) ) ) ) - cai) ;
    a = a + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau_a)))*(- ( ( ( a_inf ) ) / tau_a ) / ( ( ( ( - 1.0 ) ) ) / tau_a ) - a) ;
    b = b + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau_b)))*(- ( ( ( b_inf ) ) / tau_b ) / ( ( ( ( - 1.0 ) ) ) / tau_b ) - b) ;
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
   c_inf = 1.0 / ( 1.0 + exp ( ( _lv - theta_c ) / k_c ) ) ;
   d1_inf = 1.0 / ( 1.0 + exp ( ( _lv - theta_d1 ) / k_d1 ) ) ;
   tau_c = tau_c0 + tau_c1 / ( exp ( - ( _lv - tht_c1 ) / sig_c1 ) + exp ( - ( _lv - tht_c2 ) / sig_c2 ) ) ;
   tau_d1 = tau_d10 + tau_d11 / ( exp ( - ( _lv - tht_d11 ) / sig_d11 ) + exp ( - ( _lv - tht_d12 ) / sig_d12 ) ) ;
   a_inf = 1.0 / ( 1.0 + exp ( ( _lv - theta_a ) / k_a ) ) ;
   b_inf = 1.0 / ( 1.0 + exp ( ( _lv - theta_b ) / k_b ) ) ;
   tau_a = tau_a0 + tau_a1 / ( 1.0 + exp ( - ( _lv - tht_a ) / sig_a ) ) ;
   tau_b = tau_b0 + tau_b1 / ( exp ( - ( _lv - tht_b1 ) / sig_b1 ) + exp ( - ( _lv - tht_b2 ) / sig_b2 ) ) ;
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
 
static int  evaluate_fct2 ( _threadargsprotocomma_ double _lcai ) {
   d2_inf = 1.0 / ( 1.0 + exp ( ( _lcai - theta_d2 ) / k_d2 ) ) ;
   r_inf = 1.0 / ( 1.0 + exp ( ( _lcai - theta_r ) / k_r ) ) ;
    return 0; }
 
static void _hoc_evaluate_fct2(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 evaluate_fct2 ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 12;}
 
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
	for (_i=0; _i < 12; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 	_pv[8] = &(_ion_cai);
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
  a = a0;
  b = b0;
  c = c0;
  d2 = d20;
  d1 = d10;
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
   evaluate_fct2 ( _threadargscomma_ cai ) ;
   c = c_inf ;
   d1 = d1_inf ;
   d2 = d2_inf ;
   a = a_inf ;
   b = b_inf ;
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
   ikA = gkabar * a * a * b * ( v - ek ) ;
   ikAHP = gkcabar * ( v - ek ) * pow( r , ( power_r ) ) ;
   ik = ikD + ikA + ikAHP ;
   icaT = gcatbar * p * p * q * ( v - eca ) ;
   icaL = gcalbar * c * c * d1 * d2 * ( v - eca ) ;
   ica = icaT + icaL ;
   ilk = gl * ( v - el ) ;
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
 _slist1[5] = &(c) - _p;  _dlist1[5] = &(Dc) - _p;
 _slist1[6] = &(d1) - _p;  _dlist1[6] = &(Dd1) - _p;
 _slist1[7] = &(d2) - _p;  _dlist1[7] = &(Dd2) - _p;
 _slist1[8] = &(cai) - _p;  _dlist1[8] = &(Dcai) - _p;
 _slist1[9] = &(a) - _p;  _dlist1[9] = &(Da) - _p;
 _slist1[10] = &(b) - _p;  _dlist1[10] = &(Db) - _p;
 _slist1[11] = &(r) - _p;  _dlist1[11] = &(Dr) - _p;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "pSTN.mod";
static const char* nmodl_file_text = 
  "TITLE  STN ion channels for single compartment model\n"
  "\n"
  ":\n"
  ": Na+, K, CaT, CaL, A and AHP current \n"
  ": \n"
  "\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX stn\n"
  "	NONSPECIFIC_CURRENT ilk\n"
  "	USEION ca READ cai, cao WRITE ica, cai\n"
  "	USEION k READ ki, ko WRITE ik\n"
  "	USEION na READ nai, nao WRITE ina\n"
  "	RANGE ina, ik, ica\n"
  "	RANGE gnabar, ena, m_inf, h_inf, tau_h, tau_m		 : fast sodium\n"
  "	RANGE gkdrbar, ek, n_inf, tau_n, ikD                   : delayed K rectifier\n"
  "	RANGE gl, el, ilk                                      : leak\n"
  "	RANGE gcatbar, eca, p_inf, tau_p, q_inf, tau_q	       : T-type ca current\n"
  "	RANGE gcalbar, eca, c_inf, d1_inf, d2_inf, tau_c, tau_d1, tau_d2, icaT, icaL  : L-type ca current\n"
  "	RANGE gkabar, ek, a_inf, tau_a, b_inf, tau_b, ikA      : A-type K current\n"
  "	RANGE gkcabar, ek, r_inf, ikAHP                        : ca dependent AHP K current\n"
  "      RANGE kca, vol, caGain                                 : ca dynamics\n"
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
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	R = 8.31441 (Gas constant)\n"
  "	T 		(Absolute temp)\n"
  "	celsius		(degC)\n"
  "\n"
  ":Fast Na channel\n"
  "	gnabar   = 49e-3 (S/cm2) \n"
  "	theta_m = -40 (mV)\n"
  "	theta_h = -45.5 (mV) \n"
  "	k_m = -8 (mV)    \n"
  "	k_h = 6.4 (mV)   \n"
  "	tau_m0 = 0.2 (ms)\n"
  "	tau_m1 = 3 (ms)\n"
  "	tau_h0 = 0 (ms)\n"
  "	tau_h1 = 24.5 (ms) \n"
  "	tht_m = -53 (mV)\n"
  "	tht_h1 = -50 (mV)\n"
  "	tht_h2 = -50 (mV)\n"
  "	sig_m = -0.7 (mV)\n"
  "	sig_h1 = -15 (mV)\n"
  "	sig_h2 = 16 (mV)\n"
  "\n"
  ": Delayed rectifier K\n"
  "	gkdrbar  = 57e-3	(S/cm2)  \n"
  "	theta_n = -41 (mV)\n"
  "	k_n = -14 (mV)     \n"
  "	tau_n0 = 0 (ms)\n"
  "	tau_n1 = 11 (ms) \n"
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
  "      area\n"
  "      vol = 3.355e-11  (L) :~20um radius sphere\n"
  "      caGain = .1\n"
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
  ":Ca L current\n"
  "	gcalbar   = 15e-3 (S/cm2) \n"
  "	theta_c = -30.6 (mV)\n"
  "	theta_d1 = -60 (mV)\n"
  "	theta_d2 = 0.1e-3 (mM)\n"
  "	k_c = -5 (mV)\n"
  "	k_d1 = 7.5 (mV)\n"
  "	k_d2 = 0.02e-3 (mM)\n"
  "	tau_c0 = 45 (ms)\n"
  "	tau_c1 = 10 (ms)\n"
  "	tau_d10 = 400 (ms)\n"
  "	tau_d11 = 500 (ms)\n"
  "	tht_c1 = -27 (mV)\n"
  "	tht_c2 = -50 (mV)\n"
  "	tht_d11 = -40 (mV)\n"
  "	tht_d12 = -20 (mV)\n"
  "	sig_c1 = -20 (mV)\n"
  "	sig_c2 = 15 (mV)\n"
  "	sig_d11 = -15 (mV)\n"
  "	sig_d12 = 20 (mV)\n"
  "\n"
  "	tau_d2 = 130 (ms)\n"
  "\n"
  ":A current\n"
  "	gkabar  = 5e-3	(S/cm2)  \n"
  "	theta_a = -45 (mV)\n"
  "	theta_b = -90 (mV) \n"
  "	k_a = -14.7 (mV)    \n"
  "	k_b = 7.5 (mV)   \n"
  "	tau_a0 = 1 (ms)\n"
  "	tau_a1 = 1 (ms)\n"
  "	tau_b0 = 0 (ms)\n"
  "	tau_b1 = 200 (ms) \n"
  "	tht_a = -40 (mV)\n"
  "	tht_b1 = -60 (mV)\n"
  "	tht_b2 = -40 (mV)\n"
  "	sig_a = -0.5 (mV)\n"
  "	sig_b1 = -30 (mV)\n"
  "	sig_b2 = 10 (mV)\n"
  "\n"
  ":AHP current (Ca dependent K current)\n"
  "	gkcabar   = 1e-3 (S/cm2) \n"
  "	theta_r = 0.17e-3 (mM)\n"
  "	k_r = -0.08e-3 (mM)\n"
  "	tau_r = 2 (ms)\n"
  "	power_r = 2\n"
  "	\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	v	(mV)\n"
  "	ina	(mA/cm2)\n"
  "	ik	(mA/cm2) \n"
  "	ikD	(mA/cm2)   \n"
  "	ikA	(mA/cm2) \n"
  "	ikAHP	(mA/cm2)  \n"
  "	ica	(mA/cm2) \n"
  "	icaT	(mA/cm2) \n"
  "	icaL 	(mA/cm2)\n"
  "	ilk	(mA/cm2)\n"
  "\n"
  ":Fast Na\n"
  "	h_inf\n"
  "	tau_h	(ms)\n"
  "	m_inf\n"
  "	tau_m	(ms)\n"
  "	ena           (mV)   := 60  \n"
  "\n"
  ":Delayed rectifier\n"
  "	n_inf\n"
  "	tau_n	(ms)\n"
  "	ek         (mV) := -90\n"
  "\n"
  ":ca T current\n"
  "	p_inf\n"
  "	q_inf\n"
  "	tau_p	(ms)\n"
  "	tau_q	(ms)\n"
  "	eca           (mV)   :calc from Nernst\n"
  "\n"
  ":ca L current\n"
  "	c_inf\n"
  "	tau_c	(ms)\n"
  "	d1_inf\n"
  "	tau_d1	(ms)\n"
  "	d2_inf\n"
  "	:tau_d2	(ms)  :in PARAMETERS\n"
  "\n"
  ":A current\n"
  "	a_inf\n"
  "	tau_a	(ms)\n"
  "	b_inf\n"
  "	tau_b	(ms)\n"
  "\n"
  ":AHP (Ca dependent K current)\n"
  "	r_inf\n"
  "}\n"
  "\n"
  "STATE {\n"
  "	m h n\n"
  "	p q \n"
  "	c d1 d2\n"
  "	cai (mM) <1e-10>\n"
  "	cao (mM) <1e-10>\n"
  "	nai (mM) <1e-10>\n"
  "	nao (mM) <1e-10>\n"
  "	ki (mM) <1e-10>\n"
  "	ko (mM) <1e-10>\n"
  "	a b \n"
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
  "	:printf(\"%f %f %f\\n\", ena, ek, eca)\n"
  "\n"
  "	ina   = gnabar * m*m*m*h * (v - ena)\n"
  "	ikD   = gkdrbar * n^4 * (v - ek)\n"
  "	ikA   = gkabar * a*a*b * (v - ek)\n"
  "	ikAHP   = gkcabar * (v - ek)*r^(power_r)\n"
  "	ik=ikD+ikA+ikAHP\n"
  "	icaT   = gcatbar * p*p*q * (v - eca)\n"
  "	icaL   = gcalbar * c*c*d1*d2 * (v - eca)\n"
  "	ica=icaT+icaL\n"
  "	ilk = gl * (v - el)\n"
  "\n"
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
  "      evaluate_fct2(cai)\n"
  "	c' = (c_inf - c)/tau_c\n"
  "	d1' = (d1_inf - d1)/tau_d1\n"
  "	d2' = (d2_inf - d2)/tau_d2\n"
  "\n"
  "      :(Ica mA/cm2)*(area um2)*(1e-8 cm2/um2)*(1e-3 A/mA)*(1/(2*F) mol/C)*(1e-3 sec/msec)*(1e3 mMol/mol)(1/volume 1/L)=(mM/msec)\n"
  "	cai' = caGain*(-ica*area*1e-11/(2*FARADAY*vol) - kca*cai)\n"
  ":	cai' = -ica*area*somaAreaFrac*1e-11/(2*FARADAY*vol*shellVolFrac) + (5e-6 - cai)/kca\n"
  "\n"
  "	a' = (a_inf - a)/tau_a\n"
  "	b' = (b_inf - b)/tau_b\n"
  "\n"
  "	r' = (r_inf - r)/tau_r\n"
  "}\n"
  "\n"
  "UNITSOFF\n"
  "\n"
  "INITIAL {\n"
  "\n"
  "	evaluate_fct(v)\n"
  "	m = m_inf \n"
  "	h = h_inf   \n"
  "	n = n_inf   \n"
  "	p = p_inf \n"
  "	q = q_inf  \n"
  "\n"
  "	evaluate_fct2(cai)\n"
  "	c = c_inf \n"
  "	d1 = d1_inf  \n"
  "	d2 = d2_inf   \n"
  "\n"
  "	a = a_inf \n"
  "	b = b_inf   \n"
  "\n"
  "	r = r_inf \n"
  "}\n"
  "\n"
  "PROCEDURE evaluate_fct(v(mV)) { \n"
  ":Fast Na current\n"
  "	h_inf = 1/(1+exp((v-theta_h)/k_h))\n"
  "	m_inf = 1/(1+exp((v-theta_m)/k_m))\n"
  "	tau_h = tau_h0 + tau_h1/(exp(-(v-tht_h1)/sig_h1) + exp(-(v-tht_h2)/sig_h2)) \n"
  "	tau_m = tau_m0 + tau_m1/(1+exp(-(v-tht_m)/sig_m)) \n"
  "\n"
  ":Delayed rectifier K\n"
  "	n_inf = 1/(1+exp((v-theta_n)/k_n))\n"
  "	tau_n = tau_n0 + tau_n1/(exp(-(v-tht_n1)/sig_n1) + exp(-(v-tht_n2)/sig_n2)) \n"
  "\n"
  ":Ca T current\n"
  "	p_inf = 1/(1+exp((v-theta_p)/k_p))\n"
  "	q_inf = 1/(1+exp((v-theta_q)/k_q))\n"
  "	tau_p = tau_p0 + tau_p1/(exp(-(v-tht_p1)/sig_p1) + exp(-(v-tht_p2)/sig_p2)) \n"
  "	tau_q = tau_q0 + tau_q1/(exp(-(v-tht_q1)/sig_q1) + exp(-(v-tht_q2)/sig_q2))\n"
  "\n"
  ":Ca L current\n"
  "	c_inf = 1/(1+exp((v-theta_c)/k_c))\n"
  "	d1_inf = 1/(1+exp((v-theta_d1)/k_d1))\n"
  "	tau_c = tau_c0 + tau_c1/(exp(-(v-tht_c1)/sig_c1) + exp(-(v-tht_c2)/sig_c2))  \n"
  "	tau_d1 = tau_d10 + tau_d11/(exp(-(v-tht_d11)/sig_d11) + exp(-(v-tht_d12)/sig_d12))  \n"
  "\n"
  ":A current\n"
  "	a_inf = 1/(1+exp((v-theta_a)/k_a))\n"
  "	b_inf = 1/(1+exp((v-theta_b)/k_b))\n"
  "	tau_a = tau_a0 + tau_a1/(1+exp(-(v-tht_a)/sig_a))\n"
  "	tau_b = tau_b0 + tau_b1/(exp(-(v-tht_b1)/sig_b1) + exp(-(v-tht_b2)/sig_b2))  \n"
  "\n"
  "}\n"
  "\n"
  "PROCEDURE evaluate_fct2(cai(mM)) { \n"
  ":Ca L current\n"
  "	d2_inf = 1/(1+exp((cai-theta_d2)/k_d2))\n"
  "\n"
  ":AHP current\n"
  "	r_inf = 1/(1+exp((cai-theta_r)/k_r))\n"
  "}\n"
  "\n"
  "UNITSON\n"
  ;
#endif
