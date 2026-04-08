(function(){
  var t=localStorage.getItem('rs-theme')||'light';
  document.documentElement.setAttribute('data-theme',t);
})();
function toggleTheme(){
  var html=document.documentElement;
  var isDark=html.getAttribute('data-theme')==='dark';
  html.setAttribute('data-theme',isDark?'light':'dark');
  localStorage.setItem('rs-theme',isDark?'light':'dark');
}
