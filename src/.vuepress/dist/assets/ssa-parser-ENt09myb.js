var F=Object.defineProperty;var A=(c,t,s)=>t in c?F(c,t,{enumerable:!0,configurable:!0,writable:!0,value:s}):c[t]=s;var u=(c,t,s)=>(A(c,typeof t!="symbol"?t+"":t,s),s);import{_ as L}from"./app-a6Cj9ICh.js";import{b as P,p as R}from"./prod-LtbAT10f.js";const d=/^Format:[\s\t]*/,O=/^Style:[\s\t]*/,y=/^Dialogue:[\s\t]*/,S=/[\s\t]*,[\s\t]*/,I=/\{[^}]+\}/g,w=/\\N/g,x=/^\[(.*)[\s\t]?Styles\]$/,k=/^\[(.*)[\s\t]?Events\]$/;class V{constructor(){u(this,"h");u(this,"O",0);u(this,"c",null);u(this,"l",[]);u(this,"m",[]);u(this,"N",null);u(this,"f");u(this,"P",{})}async init(t){this.h=t,t.errors&&(this.f=(await L(()=>import("./errors-Pys1x_XQ.js"),__vite__mapDeps([0,1,2]))).ParseErrorBuilder)}parse(t,s){var e,a;if(this.O)switch(this.O){case 1:if(t==="")this.O=0;else if(O.test(t))if(this.N){const i=t.replace(O,"").split(S);this.S(i)}else this.g((e=this.f)==null?void 0:e.T("Style",s));else d.test(t)?this.N=t.replace(d,"").split(S):k.test(t)&&(this.N=null,this.O=2);break;case 2:if(t==="")this.Q();else if(y.test(t))if(this.Q(),this.N){const i=t.replace(y,"").split(S),r=this.U(i,s);r&&(this.c=r)}else this.g((a=this.f)==null?void 0:a.T("Dialogue",s));else this.c?this.c.text+=`
`+t.replace(I,"").replace(w,`
`):d.test(t)?this.N=t.replace(d,"").split(S):x.test(t)?(this.N=null,this.O=1):k.test(t)&&(this.N=null)}else t===""||(x.test(t)?(this.N=null,this.O=1):k.test(t)&&(this.N=null,this.O=2))}done(){return{metadata:{},cues:this.l,regions:[],errors:this.m}}Q(){var t,s;this.c&&(this.l.push(this.c),(s=(t=this.h).onCue)==null||s.call(t,this.c),this.c=null)}S(t){let s="Default",e={},a,i="center",r="bottom",f,o=1.2,n,p,h=3,b=[];for(let g=0;g<this.N.length;g++){const M=this.N[g],l=t[g];switch(M){case"Name":s=l;break;case"Fontname":e["font-family"]=l;break;case"Fontsize":e["font-size"]=`calc(${l} / var(--overlay-height))`;break;case"PrimaryColour":const N=T(l);N&&(e["--cue-color"]=N);break;case"BorderStyle":h=parseInt(l,10);break;case"BackColour":p=T(l);break;case"OutlineColour":const E=T(l);E&&(n=E);break;case"Bold":parseInt(l)&&(e["font-weight"]="bold");break;case"Italic":parseInt(l)&&(e["font-style"]="italic");break;case"Underline":parseInt(l)&&(e["text-decoration"]="underline");break;case"StrikeOut":parseInt(l)&&(e["text-decoration"]="line-through");break;case"Spacing":e["letter-spacing"]=l+"px";break;case"AlphaLevel":e.opacity=parseFloat(l);break;case"ScaleX":b.push(`scaleX(${parseFloat(l)/100})`);break;case"ScaleY":b.push(`scaleY(${parseFloat(l)/100})`);break;case"Angle":b.push(`rotate(${l}deg)`);break;case"Shadow":o=parseInt(l,10)*1.2;break;case"MarginL":e["--cue-width"]="auto",e["--cue-left"]=parseFloat(l)+"px";break;case"MarginR":e["--cue-width"]="auto",e["--cue-right"]=parseFloat(l)+"px";break;case"MarginV":f=parseFloat(l);break;case"Outline":a=parseInt(l,10);break;case"Alignment":const m=parseInt(l,10);switch(m>=4&&(r=m>=7?"top":"center"),m%3){case 1:i="start";break;case 2:i="center";break;case 3:i="end";break}}}if(e.R=r,e["--cue-white-space"]="normal",e["--cue-line-height"]="normal",e["--cue-text-align"]=i,r==="center"?(e["--cue-top"]="50%",b.push("translateY(-50%)")):e[`--cue-${r}`]=(f||0)+"px",h===1&&(e["--cue-padding-y"]="0"),(h===1||p)&&(e["--cue-bg-color"]=h===1?"none":p),h===3&&n&&(e["--cue-outline"]=`${a}px solid ${n}`),h===1&&typeof a=="number"){const g=p??"#000";e["--cue-text-shadow"]=[n&&_(a*1.2,o*1.2,n),n?_(a*(a/2),o*(a/2),g):_(a,o,g)].filter(Boolean).join(", ")}b.length&&(e["--cue-transform"]=b.join(" ")),this.P[s]=e}U(t,s){const e=this.V(t),a=this.q(e.Start,e.End,s);if(!a)return;const i=new P(a[0],a[1],""),r={...this.P[e.Style]||{}},f=e.Name?`<v ${e.Name}>`:"",o=r.R,n=e.MarginL&&parseFloat(e.MarginL),p=e.MarginR&&parseFloat(e.MarginR),h=e.MarginV&&parseFloat(e.MarginV);return n&&(r["--cue-width"]="auto",r["--cue-left"]=n+"px"),p&&(r["--cue-width"]="auto",r["--cue-right"]=p+"px"),h&&o!=="center"&&(r[`--cue-${o}`]=h+"px"),i.text=f+t.slice(this.N.length-1).join(", ").replace(I,"").replace(w,`
`),delete r.R,Object.keys(r).length&&(i.style=r),i}V(t){const s={};for(let e=0;e<this.N.length;e++)s[this.N[e]]=t[e];return s}q(t,s,e){var r,f,o;const a=R(t),i=R(s);if(a!==null&&i!==null&&i>a)return[a,i];a===null&&this.g((r=this.f)==null?void 0:r.s(t,e)),i===null&&this.g((f=this.f)==null?void 0:f.t(s,e)),a!=null&&i!==null&&i>a&&this.g((o=this.f)==null?void 0:o.u(a,i,e))}g(t){var s,e;if(t){if(this.m.push(t),this.h.strict)throw this.h.cancel(),t;(e=(s=this.h).onError)==null||e.call(s,t)}}}function T(c){const t=parseInt(c.replace("&H",""),16);if(t>=0){const e=(t>>24&255^255)/255,a=t>>16&255,i=t>>8&255;return"rgba("+[t&255,i,a,e].join(",")+")"}return null}function _(c,t,s){const e=Math.ceil(2*Math.PI*c);let a="";for(let i=0;i<e;i++){const r=2*Math.PI*i/e;a+=c*Math.cos(r)+"px "+t*Math.sin(r)+"px 0 "+s+(i==e-1?"":",")}return a}function Y(){return new V}export{V as SSAParser,Y as default};
function __vite__mapDeps(indexes) {
  if (!__vite__mapDeps.viteFileDeps) {
    __vite__mapDeps.viteFileDeps = ["assets/errors-Pys1x_XQ.js","assets/prod-LtbAT10f.js","assets/app-a6Cj9ICh.js"]
  }
  return indexes.map((i) => __vite__mapDeps.viteFileDeps[i])
}