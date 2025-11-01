import subprocess
import tempfile, os

import ffmpeg
import json
from PIL import Image, PngImagePlugin


def extract_audio_tracks(source_video, verbose=False, query_only=False):
    """
    Extract all audio tracks from a source video into temporary AAC files.

    Returns:
        Tuple:
          - List of temp file paths for extracted audio tracks
          - List of corresponding metadata dicts:
              {'codec', 'sample_rate', 'channels', 'duration', 'language'}
              where 'duration' is set to container duration (for consistency).
    """
    probe = ffmpeg.probe(source_video)
    audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']
    container_duration = float(probe['format'].get('duration', 0.0))

    if not audio_streams:
        if query_only: return 0
        if verbose: print(f"No audio track found in {source_video}")
        return [], []

    if query_only:
        return len(audio_streams)

    if verbose:
        print(f"Found {len(audio_streams)} audio track(s), container duration = {container_duration:.3f}s")

    file_paths = []
    metadata = []

    for i, stream in enumerate(audio_streams):
        fd, temp_path = tempfile.mkstemp(suffix=f'_track{i}.aac', prefix='audio_')
        os.close(fd)

        file_paths.append(temp_path)
        metadata.append({
            'codec': stream.get('codec_name'),
            'sample_rate': int(stream.get('sample_rate', 0)),
            'channels': int(stream.get('channels', 0)),
            'duration': container_duration,
            'language': stream.get('tags', {}).get('language', None)
        })

        ffmpeg.input(source_video).output(
            temp_path,
            **{f'map': f'0:a:{i}', 'acodec': 'aac', 'b:a': '128k'}
        ).overwrite_output().run(quiet=not verbose)

    return file_paths, metadata



def combine_and_concatenate_video_with_audio_tracks(
    save_path_tmp, video_path,
    source_audio_tracks, new_audio_tracks,
    source_audio_duration, audio_sampling_rate,
    new_audio_from_start=False,
    source_audio_metadata=None,
    audio_bitrate='128k',
    audio_codec='aac',
    verbose = False
):
    inputs, filters, maps, idx = ['-i', video_path], [], ['-map', '0:v'], 1
    metadata_args = []
    sources = source_audio_tracks or []
    news = new_audio_tracks or []

    duplicate_source = len(sources) == 1 and len(news) > 1
    N = len(news) if source_audio_duration == 0 else max(len(sources), len(news)) or 1

    for i in range(N):
        s = (sources[i] if i < len(sources)
             else sources[0] if duplicate_source else None)
        n = news[i] if len(news) == N else (news[0] if news else None)

        if source_audio_duration == 0:
            if n:
                inputs += ['-i', n]
                filters.append(f'[{idx}:a]apad=pad_dur=100[aout{i}]')
                idx += 1
            else:
                filters.append(f'anullsrc=r={audio_sampling_rate}:cl=mono,apad=pad_dur=100[aout{i}]')
        else:
            if s:
                inputs += ['-i', s]
                meta = source_audio_metadata[i] if source_audio_metadata and i < len(source_audio_metadata) else {}
                needs_filter = (
                    meta.get('codec') != audio_codec or
                    meta.get('sample_rate') != audio_sampling_rate or
                    meta.get('channels') != 1 or
                    meta.get('duration', 0) < source_audio_duration
                )
                if needs_filter:
                    filters.append(
                        f'[{idx}:a]aresample={audio_sampling_rate},aformat=channel_layouts=mono,'
                        f'apad=pad_dur={source_audio_duration},atrim=0:{source_audio_duration},asetpts=PTS-STARTPTS[s{i}]')
                else:
                    filters.append(
                        f'[{idx}:a]apad=pad_dur={source_audio_duration},atrim=0:{source_audio_duration},asetpts=PTS-STARTPTS[s{i}]')
                if lang := meta.get('language'):
                    metadata_args += ['-metadata:s:a:' + str(i), f'language={lang}']
                idx += 1
            else:
                filters.append(
                    f'anullsrc=r={audio_sampling_rate}:cl=mono,atrim=0:{source_audio_duration},asetpts=PTS-STARTPTS[s{i}]')

            if n:
                inputs += ['-i', n]
                start = '0' if new_audio_from_start else source_audio_duration
                filters.append(
                    f'[{idx}:a]aresample={audio_sampling_rate},aformat=channel_layouts=mono,'
                    f'atrim=start={start},asetpts=PTS-STARTPTS[n{i}]')
                filters.append(f'[s{i}][n{i}]concat=n=2:v=0:a=1[aout{i}]')
                idx += 1
            else:
                filters.append(f'[s{i}]apad=pad_dur=100[aout{i}]')

        maps += ['-map', f'[aout{i}]']

    cmd = ['ffmpeg', '-y', *inputs,
           '-filter_complex', ';'.join(filters),  # ✅ Only change made
           *maps, *metadata_args,
           '-c:v', 'copy',
           '-c:a', audio_codec,
           '-b:a', audio_bitrate,
           '-ar', str(audio_sampling_rate),
           '-ac', '1',
           '-shortest', save_path_tmp]

    if verbose:
        print(f"ffmpeg command: {cmd}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg error: {e.stderr}")


def combine_video_with_audio_tracks(target_video, audio_tracks, output_video,
                                     audio_metadata=None, verbose=False):
    if not audio_tracks:
        if verbose: print("No audio tracks to combine."); return False

    dur = float(next(s for s in ffmpeg.probe(target_video)['streams']
                     if s['codec_type'] == 'video')['duration'])
    if verbose: print(f"Video duration: {dur:.3f}s")

    cmd = ['ffmpeg', '-y', '-i', target_video]
    for path in audio_tracks:
        cmd += ['-i', path]

    cmd += ['-map', '0:v']
    for i in range(len(audio_tracks)):
        cmd += ['-map', f'{i+1}:a']

    for i, meta in enumerate(audio_metadata or []):
        if (lang := meta.get('language')):
            cmd += ['-metadata:s:a:' + str(i), f'language={lang}']

    cmd += ['-c:v', 'copy', '-c:a', 'copy', '-t', str(dur), output_video]

    result = subprocess.run(cmd, capture_output=not verbose, text=True)
    if result.returncode != 0:
        raise Exception(f"FFmpeg error:\n{result.stderr}")
    if verbose:
        print(f"Created {output_video} with {len(audio_tracks)} audio track(s)")
    return True


def cleanup_temp_audio_files(audio_tracks, verbose=False):
    """
    Clean up temporary audio files.
    
    Args:
        audio_tracks: List of audio file paths to delete
        verbose: Enable verbose output (default: False)
        
    Returns:
        Number of files successfully deleted
    """
    deleted_count = 0
    
    for audio_path in audio_tracks:
        try:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                deleted_count += 1
                if verbose:
                    print(f"Cleaned up {audio_path}")
        except PermissionError:
            print(f"Warning: Could not delete {audio_path} (file may be in use)")
        except Exception as e:
            print(f"Warning: Error deleting {audio_path}: {e}")
    
    if verbose and deleted_count > 0:
        print(f"Successfully deleted {deleted_count} temporary audio file(s)")
    
    return deleted_count


def _enc_uc(s):
    try: return b"ASCII\0\0\0" + s.encode("ascii")
    except UnicodeEncodeError: return b"UNICODE\0" + s.encode("utf-16le")

def _dec_uc(b):
    if not isinstance(b, (bytes, bytearray)):
        try: b = bytes(b)
        except Exception: return None
    if b.startswith(b"ASCII\0\0\0"): return b[8:].decode("ascii", "ignore")
    if b.startswith(b"UNICODE\0"):   return b[8:].decode("utf-16le", "ignore")
    return b.decode("utf-8", "ignore")

def _legacy_save_image_metadata(image_path, metadata_dict, **save_kwargs):
    j = json.dumps(metadata_dict, ensure_ascii=False)
    ext = os.path.splitext(image_path)[1].lower()
    with Image.open(image_path) as im:
        if ext == ".png":
            png_info = PngImagePlugin.PngInfo()
            png_info.add_text("comment", j)
            im.save(image_path, pnginfo=png_info, **save_kwargs)
            return True
        if ext in (".jpg", ".jpeg"):
            im.save(image_path, comment=j.encode("utf-8"), **save_kwargs)
            return True
        if ext == ".webp":
            import piexif

            exif = {
                "0th": {},
                "Exif": {piexif.ExifIFD.UserComment: _enc_uc(j)},
                "GPS": {},
                "1st": {},
                "thumbnail": None,
            }
            im.save(image_path, format="WEBP", exif=piexif.dump(exif), **save_kwargs)
            return True
        raise ValueError(f"Unsupported format for metadata embedding: {ext}")


def read_image_metadata(image_path):
    try:
        ext = os.path.splitext(image_path)[1].lower()
        with Image.open(image_path) as im:
            if ext == ".png":
                val = (getattr(im, "text", {}) or {}).get("comment") or im.info.get("comment")
                return json.loads(val) if val else None
            if ext in (".jpg", ".jpeg"):
                val = im.info.get("comment")
                if isinstance(val, (bytes, bytearray)): val = val.decode("utf-8", "ignore")
                if val:
                    try: return json.loads(val)
                    except Exception: pass
                exif = getattr(im, "getexif", lambda: None)()
                if exif:
                    uc = exif.get(37510)  # UserComment
                    s = _dec_uc(uc) if uc else None
                    if s:
                        try: return json.loads(s)
                        except Exception: pass
                return None
            if ext == ".webp":
                exif_bytes = Image.open(image_path).info.get("exif")
                if not exif_bytes: return None
                import piexif
                uc = piexif.load(exif_bytes).get("Exif", {}).get(piexif.ExifIFD.UserComment)
                s = _dec_uc(uc) if uc else None
                return json.loads(s) if s else None
            return None
    except Exception as e:
        get_notifications_logger().warning("Error reading metadata from %s: %s", image_path, e)
        return None
