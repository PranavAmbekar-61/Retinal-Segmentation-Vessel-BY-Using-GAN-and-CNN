var selectedFile = null;
var lastScanId = null;

var dropZone  = document.getElementById('drop-zone');
var fileInput = document.getElementById('file-input');

dropZone.addEventListener('dragover',  function(e){ e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', function(){ dropZone.classList.remove('dragover'); });
dropZone.addEventListener('drop', function(e){
  e.preventDefault(); dropZone.classList.remove('dragover');
  if(e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});
dropZone.addEventListener('click', function(){ fileInput.click(); });
fileInput.addEventListener('change', function(){ if(fileInput.files[0]) handleFile(fileInput.files[0]); });

function handleFile(file){
  document.getElementById('upload-error').style.display = 'none';
  if(['image/jpeg','image/png'].indexOf(file.type) === -1){
    showUploadErr('Only JPG and PNG files are accepted.'); return;
  }
  if(file.size > 10*1024*1024){
    showUploadErr('File exceeds 10 MB limit.'); return;
  }
  selectedFile = file;
  document.getElementById('preview-img').src = URL.createObjectURL(file);
  document.getElementById('preview-meta').textContent = file.name + ' · ' + fmtSize(file.size);
  var ext = file.type === 'image/jpeg' ? 'JPEG' : 'PNG';
  document.getElementById('file-details').innerHTML =
    '<div class="il-row"><span>Name</span><span>'+file.name+'</span></div>'+
    '<div class="il-row"><span>Format</span><span>'+ext+'</span></div>'+
    '<div class="il-row"><span>Size</span><span>'+fmtSize(file.size)+'</span></div>'+
    '<div class="il-row"><span>Output</span><span>4-panel result</span></div>'+
    '<div class="il-row"><span>Status</span><span style="color:var(--green);font-weight:600">Ready</span></div>';
  document.getElementById('preview-section').style.display = 'block';
  setStep(1);
}

function clearFile(){
  selectedFile = null; fileInput.value = '';
  document.getElementById('preview-section').style.display = 'none';
  document.getElementById('upload-error').style.display = 'none';
}

async function runAnalysis(){
  if(!selectedFile) return;
  showSection('processing'); setStep(2);
  var fd = new FormData();
  fd.append('image', selectedFile);
  var patientSelect = document.getElementById('patient-select');
  if(patientSelect) {
      fd.append('patient_id', patientSelect.value);
  }
  var claheToggle = document.getElementById('clahe-toggle');
  if(claheToggle) {
      fd.append('clahe', claheToggle.checked ? 'true' : 'false');
  }
  try{
    var res  = await fetch('/predict', { method:'POST', body:fd });
    var data = await res.json();
    if(!res.ok || !data.success) throw new Error(data.error || 'Segmentation failed.');

    // Set all panel images
    document.getElementById('result-4panel').src   = 'data:image/png;base64,' + data.panel_4;
    document.getElementById('p1-raw').src          = 'data:image/png;base64,' + data.p1_raw;
    document.getElementById('p2-filter').src       = 'data:image/png;base64,' + data.p2_filter;
    document.getElementById('p3-unet').src         = 'data:image/png;base64,' + data.p3_unet;
    document.getElementById('p4-gan').src          = 'data:image/png;base64,' + data.p4_gan;
    document.getElementById('result-original').src = 'data:image/png;base64,' + data.original;
    document.getElementById('overlay-img').src     = 'data:image/png;base64,' + data.overlay;

    document.getElementById('rs-time').textContent  = data.elapsed_ms;
    document.getElementById('time-chip').textContent = 'Completed in ' + data.elapsed_ms + ' ms';
    document.getElementById('rs-quality').textContent = Math.round(data.quality.score) + '/100';

    var qf = document.getElementById('quality-flags');
    if(data.quality.flags && data.quality.flags.length){
      qf.textContent = 'Image quality warning: ' + data.quality.flags.join(', ');
      qf.style.display = 'block';
    } else {
      qf.style.display = 'none';
    }

    document.getElementById('ai-clinical').textContent = data.suggestions.clinical;
    document.getElementById('ai-plain').textContent = data.suggestions.plain;
    lastScanId = data.scan_id;
    document.getElementById('meta-msg').textContent = '';
    document.getElementById('scan-notes').value = '';
    document.getElementById('scan-tags').value = '';
    document.getElementById('scan-followup').value = '';

    showSection('result'); setStep(3);
  } catch(e){
    showSection('upload'); setStep(1);
    showUploadErr('Error: ' + e.message);
    showToast(e.message, true);
  }
}

function dlPanel(imgId, filename){
  var img = document.getElementById(imgId);
  if(!img || !img.src || img.src === window.location.href) return;
  var a = document.createElement('a');
  a.href = img.src; a.download = filename; a.click();
}

function resetUpload(){
  selectedFile = null; fileInput.value = '';
  document.getElementById('preview-section').style.display = 'none';
  lastScanId = null;
  document.getElementById('scan-notes').value = '';
  document.getElementById('scan-tags').value = '';
  document.getElementById('scan-followup').value = '';
  document.getElementById('meta-msg').textContent = '';
  showSection('upload'); setStep(1);
}

function showSection(s){
  document.getElementById('section-upload').style.display     = s==='upload'     ? 'block':'none';
  document.getElementById('section-processing').style.display = s==='processing' ? 'block':'none';
  document.getElementById('section-result').style.display     = s==='result'     ? 'block':'none';
}

function setStep(n){
  for(var i=1;i<=3;i++){
    var el=document.getElementById('st'+i);
    el.classList.remove('active','done');
    if(i<n) el.classList.add('done');
    if(i===n) el.classList.add('active');
  }
}

function showUploadErr(msg){
  var e=document.getElementById('upload-error'); e.textContent=msg; e.style.display='block';
}

function showToast(msg,err){
  var t=document.getElementById('toast');
  t.textContent=msg; t.className='toast show'+(err?' toast-err':'');
  setTimeout(function(){t.classList.remove('show');},4000);
}

function fmtSize(b){
  if(b<1024) return b+' B';
  if(b<1024*1024) return (b/1024).toFixed(1)+' KB';
  return (b/1024/1024).toFixed(2)+' MB';
}

async function saveScanMeta(){
  if(!lastScanId){
    document.getElementById('meta-msg').textContent = 'No scan to save.';
    return;
  }
  var notes = document.getElementById('scan-notes').value || '';
  var tags = document.getElementById('scan-tags').value || '';
  var followUp = document.getElementById('scan-followup').value || '';
  try{
    var res = await fetch('/scan/meta/' + lastScanId, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({notes: notes, diagnosis_tags: tags, follow_up_date: followUp})
    });
    var data = await res.json();
    if(!res.ok || !data.success) throw new Error(data.error || 'Save failed.');
    document.getElementById('meta-msg').textContent = 'Saved.';
  } catch(e){
    document.getElementById('meta-msg').textContent = e.message;
  }
}
