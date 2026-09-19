/* WAF-ML SOC console theme + interactions */
(function(){
  const root=document.documentElement;
  const key="waf-theme";
  const sidebarKey="waf-sidebar";
  const button=document.getElementById("themeToggle");
  const sidebarToggle=document.getElementById("sidebarToggle");
  const sidebarIcon=document.getElementById("sidebarToggleIcon");

  try{
    const saved=localStorage.getItem(key);
    if(saved==="light" || saved==="dark") root.dataset.theme=saved;
    const sidebarState=localStorage.getItem(sidebarKey);
    if(sidebarState==="collapsed" || sidebarState==="expanded") root.dataset.sidebar=sidebarState;
  }catch(e){}

  function syncTheme(){
    if(!button) return;
    const light=root.dataset.theme==="light";
    const sun=button.querySelector(".theme-toggle-icon--sun");
    const moon=button.querySelector(".theme-toggle-icon--moon");
    if(sun) sun.style.display=light?"inline":"none";
    if(moon) moon.style.display=light?"none":"inline";
    button.title=light?"Switch to dark mode":"Switch to light mode";
    button.setAttribute("aria-label",button.title);
    button.setAttribute("aria-pressed",light?"true":"false");
  }

  function syncSidebar(){
    if(!sidebarToggle) return;
    const collapsed=root.dataset.sidebar==="collapsed";
    if(sidebarIcon) sidebarIcon.textContent=collapsed?"›":"‹";
    sidebarToggle.title=collapsed?"Open sidebar":"Collapse sidebar";
    sidebarToggle.setAttribute("aria-label",sidebarToggle.title);
    sidebarToggle.setAttribute("aria-expanded",collapsed?"false":"true");
  }

  if(button){
    syncTheme();
    button.addEventListener("click",()=>{
      const light=root.dataset.theme==="light";
      root.dataset.theme=light?"dark":"light";
      try{localStorage.setItem(key,root.dataset.theme);}catch(e){}
      syncTheme();
    });
  }

  if(sidebarToggle){
    syncSidebar();
    sidebarToggle.addEventListener("click",()=>{
      const collapsed=root.dataset.sidebar==="collapsed";
      root.dataset.sidebar=collapsed?"expanded":"collapsed";
      try{localStorage.setItem(sidebarKey,root.dataset.sidebar);}catch(e){}
      syncSidebar();
    });
  }
})();

document.addEventListener("DOMContentLoaded",()=>{
  const root=document.documentElement;
  const key="waf-theme";
  const button=document.getElementById("themeToggle");
  try{
    const saved=localStorage.getItem(key);
    if(saved==="light" || saved==="dark") root.dataset.theme=saved;
  }catch(e){}
  function syncTheme(){
    if(!button) return;
    const light=root.dataset.theme==="light";
    const sun=button.querySelector(".theme-toggle-icon--sun");
    const moon=button.querySelector(".theme-toggle-icon--moon");
    if(sun) sun.style.display=light?"inline":"none";
    if(moon) moon.style.display=light?"none":"inline";
    button.title=light?"Switch to dark mode":"Switch to light mode";
    button.setAttribute("aria-label",button.title);
    button.setAttribute("aria-pressed",light?"true":"false");
  }
  if(button){
    syncTheme();
    button.addEventListener("click",()=>{
      const light=root.dataset.theme==="light";
      root.dataset.theme=light?"dark":"light";
      try{localStorage.setItem(key,root.dataset.theme);}catch(e){}
      syncTheme();
    });
  }
})();

document.addEventListener("DOMContentLoaded",()=>{
  const path=window.location.pathname;
  document.querySelectorAll(".side-link").forEach(link=>{
    if(link.getAttribute("href")===path) link.classList.add("active");
  });
  const clock=document.getElementById("topbarTime");
  const tick=()=>{if(clock) clock.textContent=new Date().toLocaleTimeString([], {hour12:false});};
  tick(); setInterval(tick,1000);
  document.querySelectorAll(".stat-value").forEach(el=>{
    const raw=parseInt(el.textContent.replace(/[^0-9]/g,""),10);
    if(!Number.isNaN(raw)&&raw>0){
      let start=0; const duration=700; const step=Math.max(raw/(duration/16),1);
      const unit=el.querySelector(".stat-unit"); const unitText=unit?unit.outerHTML:"";
      const timer=setInterval(()=>{start=Math.min(start+step,raw);el.innerHTML=Math.floor(start).toLocaleString()+unitText;if(start>=raw)clearInterval(timer)},16);
    }
  });
  document.querySelectorAll(".breakdown-bar").forEach(bar=>{
    const target=bar.style.width;bar.style.width="0";
    requestAnimationFrame(()=>setTimeout(()=>bar.style.width=target,120));
  });
});
