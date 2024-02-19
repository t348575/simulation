use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, GenericParam};

#[proc_macro_derive(Name)]
pub fn derive_neuron_name(input: TokenStream) -> TokenStream {
    let input: DeriveInput = parse_macro_input!(input);
    let name_str = input.ident.to_string();
    let name = input.ident;

    let generic_names = input
        .generics
        .params
        .iter()
        .filter_map(|x| {
            if let GenericParam::Type(t) = x {
                Some(t)
            } else {
                None
            }
        })
        .map(|x| x.ident.clone())
        .collect::<Vec<_>>();
    let generics = input.generics;
    quote! {
        impl #generics NeuronName for #name<#(#generic_names, )*> {
            fn name(&self) -> &'static str {
                #name_str
            }
        }
    }
    .into()
}

#[proc_macro_derive(SubTraits)]
pub fn derive_neuron_sub_traits(input: TokenStream) -> TokenStream {
    let input: DeriveInput = parse_macro_input!(input);
    let name = input.ident;

    let generic_names = input
        .generics
        .params
        .iter()
        .filter_map(|x| {
            if let GenericParam::Type(t) = x {
                Some(t)
            } else {
                None
            }
        })
        .map(|x| x.ident.clone())
        .collect::<Vec<_>>();
    let generics = input.generics;
    quote! {
        impl #generics NeuronSubTraits for #name<#(#generic_names, )*> {}
    }
    .into()
}
