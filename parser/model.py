import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from parser.modules import BiLSTM, CharLSTM, CRF, Biaffine
from parser.modules.dropout import IndependentDropout, SharedDropout


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.pretrained = False

        # the embedding layer
        self.word_embed = nn.Embedding(num_embeddings=args.n_words,
                                       embedding_dim=args.n_embed)
        n_lstm_input = args.n_embed

        # feat_embed
        if self.args.feat == 'char':
            self.char_embed = CharLSTM(n_chars=args.n_chars, n_embed=args.n_char_embed, n_out=args.n_embed)
            n_lstm_input += args.n_embed
        self.embed_dropout = IndependentDropout(p=args.embed_dropout)

        # the lstm layer
        self.lstm = BiLSTM(input_size=n_lstm_input,
                           hidden_size=args.n_lstm_hidden,
                           num_layers=args.n_lstm_layers,
                           dropout=args.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=args.lstm_dropout)

        # the MLP layers
        self.biaffine = Biaffine(n_in=args.n_lstm_hidden * 2, n_out=args.n_labels,
                                 bias_x=True, bias_y=True)

        # crf
        self.crf = CRF(args.n_labels, self.args.label_bos_index, self.args.label_pad_index)

        self.pad_index = args.pad_index
        self.unk_index = args.unk_index

    def load_pretrained(self, embed):
        """
        load pretrained embedding

        Args:
            embed:

        Returns:

        """
        # word embed
        if embed is not None:
            self.word_pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)
            self.pretrained = True
        return self

    def forward(self, feed_dict):
        """

        Args:
            feed_dict:

        Returns:

        """
        # words: [batch_size, seq_len]
        words = feed_dict["words"]
        batch_size, seq_len = words.shape

        # mask: [batch_size, seq_len]
        mask = words.ne(self.pad_index)
        # lens: [batch_size]
        lens = mask.sum(dim=1)

        # set the indices larger than num_embeddings to unk_index
        ext_words = words
        if self.pretrained:
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # word embed
        word_embed = self.word_embed(ext_words)
        if self.pretrained:
            word_embed += self.word_pretrained(words)

        # feat embed
        if self.args.feat == 'char':
            chars = feed_dict["chars"]
            char_embed = self.char_embed(chars)
            word_embed, char_embed = self.embed_dropout(word_embed, char_embed)
            embed = torch.cat((word_embed, char_embed), dim=-1)
        else:
            embed = self.embed_dropout(word_embed)[0]

        # lstm
        x = pack_padded_sequence(embed, lens, True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        # pre, now: [batch_size, seq_len - 1, n_lstm_hidden * 2]
        pre = x[:, :-1]
        now = x[:, 1:]

        # emits: [batch_size, seq_len - 1, n_labels]
        emits = self.biaffine(now, pre)

        return emits

    def loss(self, emits, labels, mask):
        """

        Args:
            emits:
            labels:
            mask:

        Returns:

        """
        return self.crf(emits, labels, mask)

    @classmethod
    def load(cls, path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path, map_location=device)
        model = cls(state['args'])
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(device)

        return model

    def save(self, path):
        state_dict, pretrained = self.state_dict(), None
        if self.pretrained:
            pretrained = state_dict.pop('word_pretrained.weight')
        state = {
            'args': self.args,
            'state_dict': state_dict,
            'pretrained': pretrained
        }
        torch.save(state, path)

    def predict(self, emits, mask):
        """

        Args:
            emits:
            mask:

        Returns:

        """
        return self.crf.viterbi(emits, mask)
