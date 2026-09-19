/* WAF-ML SOC console interactions */
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
