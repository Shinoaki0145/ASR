() => {
  const player = document.createElement('audio');
  const retryButton=()=>document.getElementById('web-asr-retry');
  player.id = 'web-asr-player';
  let commandId = 0, generation = -1, epoch = 0, muted = false, active = null, ready = false;
  const acks = [];
  let stream=null, context=null, source=null, processor=null, mic=false, chunks=[], samples=0, micError=null, sequence=0, recordingId=0, captured=0;
  async function toggleMic() {
    if (mic) {
      mic=false;processor?.disconnect();source?.disconnect();
      stream?.getTracks().forEach(t=>t.stop());await context?.close();
      return false;
    }
    try {
      stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
      context=new AudioContext();await context.resume();
      // Own capture graph, no Gradio private DOM or cumulative recording blobs.
      const code=`class Capture extends AudioWorkletProcessor {process(inputs) {if(inputs[0]?.[0]) {const channels=inputs[0]; const mono=new Float32Array(channels[0].length);for(const c of channels)for(let i=0;i<c.length;i++)mono[i]+=c[i]/channels.length;this.port.postMessage(mono);}return true;}}registerProcessor('capture',Capture);`;
      const url=URL.createObjectURL(new Blob([code],{type:'application/javascript'}));
      try {await context.audioWorklet.addModule(url);} finally {URL.revokeObjectURL(url);}
      source=context.createMediaStreamSource(stream);
      processor=new AudioWorkletNode(context,'capture');
      processor.port.onmessage=({data})=>{
        if(!mic)return;
        if(samples+data.length>context.sampleRate*3){micError='Audio buffer đầy; dừng mic để tránh mất dữ liệu';toggleMic();return;}
        chunks.push(data);samples+=data.length;captured+=data.length;
      };
      source.connect(processor);processor.connect(context.destination);mic=true;recordingId++;micError=null;return true;
    } catch(e) {
      micError=e.name+': '+e.message;stream?.getTracks().forEach(t=>t.stop());await context?.close();mic=false;return false;
    }
  }
  function drain(force=false) {
    if (!ready && !force) return '';
    ready = true;
    let data='';const rate=context?.sampleRate||16000;
    if(samples){const bytes=new Uint8Array(samples*2);const view=new DataView(bytes.buffer);let offset=0;
      for(const c of chunks)for(const x of c){view.setInt16(offset,Math.round(Math.max(-1,Math.min(1,x))*32767),true);offset+=2;}
      for(let i=0;i<bytes.length;i+=8192)data+=String.fromCharCode(...bytes.subarray(i,i+8192));
    }
    chunks=[];samples=0;const err=micError;micError=null;
    return JSON.stringify({sequence:++sequence,recording_id:recordingId,rate,audio:btoa(data),mic,error:err,acks:acks.splice(0)});
  }
  function ack(c, status, extra={}) { acks.push({...c, audio_url: undefined, status, ...extra}); }
  function mediaMeta() {
    const dur = Number.isFinite(player.duration) ? Math.round(player.duration * 1000) / 1000 : null;
    const pos = Number.isFinite(player.currentTime) ? Math.round(player.currentTime * 1000) / 1000 : null;
    return {audio_duration_s: dur, played_s: pos};
  }
  function stop(status='stopped') {
    const meta = mediaMeta();
    player.pause(); player.removeAttribute('src'); player.load();
    if (active) ack(active, status, meta);
    active = null;
    if(retryButton())retryButton().style.display='none';
  }
  window.webASR = {
    toggleMic, drain, stats:()=>({mic,samples,captured,state:context?.state,rate:context?.sampleRate}),
    mute(value) { epoch++;muted = value; if (value) stop('muted'); },
    apply(snapshot) {
      if (!snapshot) return null;
      ready = true;
      document.getElementById('web-asr-audio-slot')?.append(player);
      if(retryButton())retryButton().onclick=()=>{if(active){player.play().catch(()=>{});}};
      if(snapshot.capture_error && mic){micError=snapshot.capture_error;toggleMic();}
      if (snapshot.audio_epoch < epoch) return null;
      const c = snapshot.command;
      if (snapshot.generation > generation || snapshot.audio_epoch > epoch) stop();
      generation = Math.max(generation, snapshot.generation);
      epoch = Math.max(epoch, snapshot.audio_epoch);
      muted = !snapshot.speaker;
      if (c && c.command_id > commandId) {
        commandId = c.command_id;
        if (c.kind === 'stop') {stop();ack(c,'stopped');}
        else if (c.generation === generation && c.audio_epoch === epoch && !muted) {
          stop(); active = c; player.src = c.audio_url;
          player.onplay = () => {if(active===c){ack(c,'started',mediaMeta());if(retryButton())retryButton().style.display='none';}};
          player.onended = () => { if (active === c) {ack(c, 'ended', mediaMeta()); active=null;} };
          player.onerror = () => { if (active === c) {ack(c, 'error', mediaMeta());active=null;} };
          player.play().catch(() => {if(active===c){ack(c,'blocked',mediaMeta());if(retryButton())retryButton().style.display='inline-block';}});
        }
      }
      return null;
    },
    player
  };
}
