import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2Processor
from torch.nn import CTCLoss
import torch
import torchaudio

from Files import Files
from Method import Method


class DetectStuttering:
    def __init__(self, audio_file: str = "", proper_transcript: str=""):
        self.proper_transcript = proper_transcript
        self.audio_file = audio_file


    def transcribe_detect_stuttering(self):
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        print(self.proper_transcript)
        if self.proper_transcript == "":
            self.proper_transcript = self.target_transcript().upper()
        else:
            self.proper_transcript = self.proper_transcript.upper()
        input_audio, _ = librosa.load(self.audio_file, sr=16000)
        input_features = tokenizer(input_audio, return_tensors="pt").input_values
        with torch.no_grad():
            logits = model(input_features).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)[0]
        stuttering_indices = []
        for i, (char, proper_char) in enumerate(zip(transcription, self.proper_transcript)):
            if char != proper_char:
                stuttering_indices.append(i)
        stuttering_segments = []
        if stuttering_indices:
            current_segment = [stuttering_indices[0]]
            for i in range(1, len(stuttering_indices)):
                if stuttering_indices[i] - stuttering_indices[i - 1] > 1:
                    stuttering_segments.append(current_segment)
                    current_segment = [stuttering_indices[i]]
                else:
                    current_segment.append(stuttering_indices[i])
            stuttering_segments.append(current_segment)
        stuttered_chars = self.find_sequences_not_in_common(transcription,self.proper_transcript)
        proper_transcript_length = len(self.proper_transcript)
        total_stuttered_chars = 0
        for segment in stuttered_chars:
            total_stuttered_chars = total_stuttered_chars + len(segment)
        stuttering_percentage = (total_stuttered_chars / proper_transcript_length) * 100
        stuttering_percentage = round(stuttering_percentage)
        full_text = ""
        full_text += "Stuttering Percentage: " + str(stuttering_percentage) + "%" + "\n\n"
        if stuttering_percentage<5:
            full_text += "Stuttering Severity : TRIVIAL  " + "\n\n"
        elif stuttering_percentage>=5 and stuttering_percentage<10:
            full_text += "Stuttering Severity : TRIVAL to INTERMEDIATE " + "\n\n"
        elif stuttering_percentage>=10 and stuttering_percentage<15:
            full_text += "Stuttering Severity : INTERMEDIATE  " + "\n\n"
        elif stuttering_percentage>=15 and stuttering_percentage<20:
            full_text += "Stuttering Severity : INTERMEDIATE to ACUTE " + "\n\n"
        else:
            full_text += "Stuttering Severity : ACUTE " + "\n\n"
        full_text += "Stuttered Characters : ["
        for i in range(len(stuttered_chars)):
            if i != len(stuttered_chars) - 1:
                full_text += " " + stuttered_chars[i] + ", "
            else:
                full_text += " " + stuttered_chars[i] + "] " + "\n\n"
        full_text += "Original/Target Transcript : " + transcription + "\n"
        full_text += "Audio Transcript : " + self.proper_transcript + "\n"
        return full_text


    def find_max_common_characters(self,transcription1, transcript2):
        m = len(transcription1)
        n = len(transcript2)
        lcs_matrix = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if transcription1[i - 1] == transcript2[j - 1]:
                    lcs_matrix[i][j] = lcs_matrix[i - 1][j - 1] + 1
                else:
                    lcs_matrix[i][j] = max(lcs_matrix[i - 1][j], lcs_matrix[i][j - 1])
        lcs_characters = []
        i = m
        j = n
        while i > 0 and j > 0:
            if transcription1[i - 1] == transcript2[j - 1]:
                lcs_characters.append(transcription1[i - 1])
                i -= 1
                j -= 1
            elif lcs_matrix[i - 1][j] > lcs_matrix[i][j - 1]:
                i -= 1
            else:
                j -= 1

        lcs_characters.reverse()
        lcs_string = ''.join(lcs_characters)

        return lcs_string



    def find_sequences_not_in_common(self,transcription1,proper_transcript):
        common_characters = self.find_max_common_characters(transcription1, proper_transcript)
        sequences = []
        sequence = ""
        i = 0
        j = 0

        while i < len(transcription1) and j < len(common_characters):
            if transcription1[i] == common_characters[j]:
                if sequence:
                    sequences.append(sequence)
                    sequence = ""
                i += 1
                j += 1
            else:
                sequence += transcription1[i]
                i += 1

        if sequence:
            sequences.append(sequence)

        return sequences


    def stuttering_seconds(self):
        if self.proper_transcript == "":
            self.proper_transcript = self.target_transcript().upper()
        else:
            self.proper_transcript = self.proper_transcript.upper()
        waveform, sample_rate = torchaudio.load(self.audio_file)
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        input_values = processor(waveform[0], return_tensors="pt").input_values  # Modified this line
        with torch.no_grad():
            logits = model(input_values).logits
        tokens = processor.tokenizer(self.proper_transcript, return_tensors="pt", padding=True, truncation=True)
        target_ids = tokens.input_ids
        log_probs = torch.log_softmax(logits, dim=-1)
        input_lengths = torch.tensor([log_probs.shape[1]], dtype=torch.long)
        target_lengths = torch.tensor([target_ids.shape[1]], dtype=torch.long)
        ctc_loss = CTCLoss(blank=model.config.pad_token_id)
        loss = ctc_loss(log_probs.transpose(0, 1), targets=target_ids, input_lengths=input_lengths,
                        target_lengths=target_lengths)
        model_name = 'facebook/wav2vec2-large-960h-lv60-self'
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
        input_audio, sample_rate = librosa.load(self.audio_file, sr=16000)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            input_audio = resampler(input_audio)
        input_features = tokenizer(input_audio.squeeze(), return_tensors='pt').input_values
        with torch.no_grad():
            logits = model(input_features).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        blank_token_id = tokenizer.pad_token_id
        stuttering_seconds = []
        prev_token = blank_token_id
        frame_shift = 0.02
        audio_duration = len(input_audio) / sample_rate
        for frame_idx, token_id in enumerate(predicted_ids[0]):
            if token_id != blank_token_id and token_id != prev_token:
                start_frame = frame_idx
                end_frame = frame_idx + token_id.item() - 1
                start_second = min(start_frame * frame_shift, audio_duration)
                end_second = min(end_frame * frame_shift, audio_duration)
                if end_second - start_second > 0.3:
                    stuttering_seconds.append((round(start_second, 2), round(end_second, 2)))
            prev_token = token_id
        return round(loss.item(), 2), stuttering_seconds


    def target_transcript(self):
        processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        waveform, sample_rate = torchaudio.load(self.audio_file)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        input_values = processor(waveform[0], return_tensors="pt").input_values
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_sentences = processor.batch_decode(predicted_ids)
        return predicted_sentences[0]

    def predict_megamgem(self):
        concatenated_vector_get = Method(self.audio_file).get_concatenated_vector_get()
        prediction = Files("classification_model.h5").predict_with_model(concatenated_vector_get)
        if (prediction[0] == 1):
            self.proper_transcript = self.proper_transcript.upper()
            loss, stuttering_seconds_get = self.stuttering_seconds()
            stuttering_seconds_str = ', '.join(str(sec) for sec in stuttering_seconds_get)
            full_text = self.transcribe_detect_stuttering()
            full_text = "Stuttering\nStuttering seconds: " + stuttering_seconds_str + "\n\n" + full_text
            #full_text = full_text.lower()
            return full_text
        else:
            return "Not Stuttering"





