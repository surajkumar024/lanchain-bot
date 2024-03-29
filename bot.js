import {PDFLoader} from 'langchain/document_loaders/fs/pdf';
import {RecursiveCharacterTextSplitter} from 'langchain/text_splitter';
import {OpenAIEmbeddings} from 'langchain/embeddings/openai';
import {HNSWLib} from 'langchain/vectorstores/hnswlib';
import {OpenAI} from 'langchain/llms/openai';
import {RetrievalQAChain} from 'langchain/chains';
// import {H} from 'langchain/llms/hf'
// import { HuggingFaceInferenceEmbeddings } from "langchain/embeddings/hf";
// import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import * as dotenv from 'dotenv'
dotenv.config()


const loader = new PDFLoader("./training-data/node-js-2.pdf",{
    parsedItemSeparator:""
});
const docs = await loader.load();
// console.log(docs);
/// this is the spliter function ...
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize:1000,
    chunkOverlap:90
});
// creating chunks from pdf
const splitDocs = await textSplitter.splitDocuments(docs);
const embeddings = new OpenAIEmbeddings(); // OPENAPI

console.log('emandings :: ' , embeddings)


// const embeddings = new HuggingFaceInferenceEmbeddings();

// const vectorStore = await HNSWLib.fromDocuments(
//     splitDocs, embeddings
// )

// console.log('vector store :: ' , vectorStore)

// // return 'he'

// const vectoreStoreRetriver = vectorStore.asRetriever();

// const model = new OpenAI({
//     modelName:'gpt-3.5-turbo'
// }); //// FOR OPEN_API MODELS

// // const model = new HuggingFaceTransformersEmbeddings({
// //     modelName: "hkunlp/instructor-large",
// //   });


// // const res = await model.embedQuery(
// //     "what is node js?"
// //   );

// //   console.log({ res });

// const chain = RetrievalQAChain.fromLLM(model, vectoreStoreRetriver);

// const question = 'what is node js?';
// const answer = await chain.call({
//     query:question
// })

// console.log(
//     question,
//     answer
// )